from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import MISSING, asdict, fields, is_dataclass
from datetime import time
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import yaml

from instrumentation.src.event_contract import (
    EVENT_DIRS,
    event_priority,
    event_schema_version,
    event_scope,
)
from instrumentation.src.event_envelope import LINEAGE_ENVELOPE_FIELDS

from .deployment_metadata import DEFAULT_CONTRACT_PATH, TELEMETRY_SCHEMA_VERSION
from .hashing import canonical_json_hash, file_sha256
from .portfolio import PortfolioPolicyConfig


CONTRACT_SCHEMA_VERSION = "k_stock_olr_kalcb_strategy_plugin_contract_v1"
CONTRACT_GENERATOR_VERSION = "olr_kalcb_bridge_contract_generator_v1"

KALCB_CONFIG_PATH = Path("config/kalcb.yaml")
OLR_CONFIG_MODULE_PATH = Path("strategy_olr/config.py")
KALCB_CONFIG_MODULE_PATH = Path("strategy_kalcb/config.py")
SECTOR_MAP_PATH = Path("config/olr/sector_map.yaml")
UNIVERSE_PATH = Path("config/olr_kalcb/olr_deployment_universe_103.yaml")
PORTFOLIO_POLICY_PATH = Path("config/olr_kalcb/portfolio_policy.conservative.json")
OMS_CONFIG_PATH = Path("config/oms_config.yaml")

REQUIRED_PAYLOAD_IDENTITY_FIELDS = (
    "bot_id",
    "strategy_id",
    "family_id",
    "portfolio_id",
    "account_alias",
    "deployment_id",
    "config_version",
    "code_sha",
)

CONDITIONAL_PAYLOAD_IDENTITY_FIELDS = (
    "assistant_strategy_id",
)

REQUIRED_JOIN_FIELDS = (
    "decision_id",
    "decision_ref",
    "event_ref",
    "action_ref",
    "intent_id",
    "idempotency_key",
    "portfolio_rule_event_id",
    "risk_decision_id",
    "order_id",
    "client_order_id",
    "oms_order_id",
    "kis_order_id",
    "kis_order_date",
    "kis_exec_id",
    "fill_id",
    "trade_id",
)

SNAPSHOT_EVENT_REQUIREMENTS = {
    "deployment": (
        "deployment_id",
        "strategy_ids",
        "strategy_version",
        "config_version",
        "portfolio_config_version",
        "risk_config_version",
        "allocation_version",
        "strategy_registry_version",
        "strategy_plugin_contract_path",
        "strategy_plugin_contract_hash",
        "kis_resource_plan_hash",
    ),
    "config_snapshot": (
        "deployment_id",
        "strategy_configs",
        "portfolio_policy_config",
        "risk_config",
        "sector_map_hash",
        "staged_artifacts",
        "kis_resource_plan_path",
    ),
    "resource_plan": (
        "plan_hash",
        "mode",
        "trade_date",
        "strategy_ids",
        "limit_profile",
        "lease_windows",
        "route_table",
    ),
}

RUNTIME_RESTART = "requires artifact regeneration, preflight, and runtime restart before execution"
OMS_RESTART = "requires OMS config reload or service restart before execution"

RISK_CONFIG_FIELDS = (
    "daily_loss_warn_pct",
    "daily_loss_halt_pct",
    "max_gross_exposure_pct",
    "max_net_exposure_pct",
    "max_position_pct",
    "max_positions_count",
    "max_sector_pct",
    "unknown_sector_policy",
    "strategy_budgets",
    "max_spread_bps",
    "vi_cooldown_sec",
    "regime_exposure_caps",
    "current_regime",
    "require_durable_stops",
    "default_stop_protection_mode",
    "allow_synthetic_stop_only",
    "stop_price_stale_after_sec",
    "stop_watcher_interval_sec",
    "stop_exit_order_type",
    "stop_protection_emergency_override",
)

RISK_CONFIG_DEFAULTS = {
    "daily_loss_warn_pct": 0.02,
    "daily_loss_halt_pct": 0.03,
    "max_gross_exposure_pct": 0.80,
    "max_net_exposure_pct": 0.60,
    "max_position_pct": 0.15,
    "max_positions_count": 10,
    "max_sector_pct": 0.30,
    "unknown_sector_policy": "allow",
    "strategy_budgets": {
        "PCIM": {
            "max_positions": 8,
            "max_risk_pct": 0.10,
            "capital_allocation_pct": 1.0,
        },
    },
    "max_spread_bps": 50.0,
    "vi_cooldown_sec": 600.0,
    "regime_exposure_caps": {
        "CRISIS": 0.20,
        "WEAK": 0.50,
        "NORMAL": 0.80,
        "STRONG": 1.00,
    },
    "current_regime": "NORMAL",
    "require_durable_stops": True,
    "default_stop_protection_mode": "oms_watcher",
    "allow_synthetic_stop_only": False,
    "stop_price_stale_after_sec": 30.0,
    "stop_watcher_interval_sec": 5.0,
    "stop_exit_order_type": "MARKET",
    "stop_protection_emergency_override": False,
}

SAFETY_KEYWORDS = (
    "risk",
    "stop",
    "budget",
    "position",
    "exposure",
    "leverage",
    "notional",
    "ws_",
    "rest_",
    "universe",
    "sector",
    "auction",
    "fill_timing",
    "execution",
    "portfolio",
    "allocation",
    "strategy_id",
    "timeframe",
    "entry_mode",
    "exit_mode",
)


def build_strategy_plugin_contract(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build the assistant bridge contract from executable OLR/KALCB sources."""

    root = Path(repo_root or Path.cwd()).resolve()
    kalcb_module = _load_module(root / KALCB_CONFIG_MODULE_PATH, "_bridge_kalcb_config")
    olr_module = _load_module(root / OLR_CONFIG_MODULE_PATH, "_bridge_olr_config")
    kalcb_file_config = _load_yaml(root / KALCB_CONFIG_PATH)
    kalcb_config = kalcb_module.KALCBConfig.from_mapping(kalcb_file_config)
    olr_config = olr_module.OLRConfig()
    portfolio_policy_payload = _load_json(root / PORTFOLIO_POLICY_PATH)
    portfolio_policy = PortfolioPolicyConfig(**_portable_policy_kwargs(portfolio_policy_payload))
    universe = _load_yaml(root / UNIVERSE_PATH)
    sector_map = _load_sector_map(root / SECTOR_MAP_PATH)
    oms_config = _load_yaml(root / OMS_CONFIG_PATH)
    risk_config = _effective_oms_risk_config_payload(oms_config)

    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "generator_version": CONTRACT_GENERATOR_VERSION,
        "bot_id": "k_stock_trader",
        "portfolio_id": "olr_kalcb",
        "family_id": "krx_equity",
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "strategy_ids": ["KALCB", "OLR"],
        "assistant_strategy_ids": {"KALCB": "KALCB", "OLR": "OLR"},
        "runtime": {
            "entrypoint": "deployment.olr_kalcb.runtime:prepare_runtime_session",
            "mode_gate_sequence": [
                "artifact_only_stage1",
                "artifact_only",
                "dry_run",
                "paper",
                "live",
            ],
            "strategy_plugin_contract_path": DEFAULT_CONTRACT_PATH.as_posix(),
            "deployment_metadata_output": {
                "artifact_name": "deployment_metadata.json",
                "emitted_by": "deployment.olr_kalcb.deployment_metadata.emit_deployment_metadata",
                "operator_argument": "--deployment-metadata-json",
                "environment_variable": "OLR_KALCB_DEPLOYMENT_METADATA_PATH",
                "provenance_policy": "runtime artifact only; requires real remote, full commit sha, and clean worktree",
            },
        },
        "source_artifacts": _source_artifacts(root),
        "strategies": {
            "KALCB": {
                "core_version": str(kalcb_module.KALCB_CORE_VERSION),
                "style": "intraday_breakout",
                "config_source": KALCB_CONFIG_PATH.as_posix(),
                "config_module": KALCB_CONFIG_MODULE_PATH.as_posix(),
                "editable_parameters": _strategy_parameter_registry(
                    strategy_id="KALCB",
                    config=kalcb_config,
                    aliases=kalcb_module._ALIASES,
                    config_source=KALCB_CONFIG_PATH,
                    module_source=KALCB_CONFIG_MODULE_PATH,
                    supported_entry_plan_modes=kalcb_module.SUPPORTED_ENTRY_PLAN_MODES,
                ),
            },
            "OLR": {
                "core_version": str(olr_module.OLR_CORE_VERSION),
                "style": "overnight_leader_rotation",
                "config_source": OLR_CONFIG_MODULE_PATH.as_posix(),
                "editable_parameters": _strategy_parameter_registry(
                    strategy_id="OLR",
                    config=olr_config,
                    aliases=olr_module._ALIASES,
                    config_source=OLR_CONFIG_MODULE_PATH,
                    module_source=OLR_CONFIG_MODULE_PATH,
                ),
            },
        },
        "shared_editable_resources": {
            "sector_map": _sector_map_contract(sector_map),
            "deployment_universe": _universe_contract(universe),
            "portfolio_policy": _portfolio_policy_contract(portfolio_policy),
            "oms_risk_policy": _oms_policy_contract(oms_config, risk_config),
        },
        "assistant_bridge": {
            "event_envelope": {
                "canonical_schema_version": "assistant_event_v1",
                "lineage_envelope_fields": list(LINEAGE_ENVELOPE_FIELDS),
                "required_payload_identity_fields": list(REQUIRED_PAYLOAD_IDENTITY_FIELDS),
                "conditional_payload_identity_fields": {
                    field: "required only when source strategy_id differs from the assistant profile id"
                    for field in CONDITIONAL_PAYLOAD_IDENTITY_FIELDS
                },
                "required_join_fields": list(REQUIRED_JOIN_FIELDS),
                "payload_identity_policy": "duplicate identity and joins in payload even though relay envelopes also carry them",
            },
            "event_streams": _event_stream_contract(),
            "required_snapshots": {
                key: {"event_type": key, "required_payload_fields": list(fields)}
                for key, fields in SNAPSHOT_EVENT_REQUIREMENTS.items()
            },
        },
    }
    contract["contract_hash"] = canonical_json_hash(_contract_hash_payload(contract))
    return contract


def write_strategy_plugin_contract(
    output_path: str | Path = DEFAULT_CONTRACT_PATH,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd()).resolve()
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    contract = build_strategy_plugin_contract(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _strategy_parameter_registry(
    *,
    strategy_id: str,
    config: Any,
    aliases: Mapping[str, str],
    config_source: Path,
    module_source: Path,
    supported_entry_plan_modes: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    alias_by_field: dict[str, list[str]] = defaultdict(list)
    for alias, field_name in aliases.items():
        alias_by_field[str(field_name)].append(str(alias))
    entries = []
    for field in fields(config):
        value = getattr(config, field.name)
        paths = sorted(set(alias_by_field.get(field.name) or [field.name]))
        entries.append(
            {
                "parameter_id": f"{strategy_id}.{field.name}",
                "strategy_id": strategy_id,
                "source_files": sorted(
                    {
                        config_source.as_posix(),
                        module_source.as_posix(),
                    }
                ),
                "canonical_field": field.name,
                "editable_paths": paths,
                "accepted_direct_key": field.name,
                "default_value": _jsonable(_field_default(field)),
                "effective_value": _jsonable(value),
                "value_type": _value_type(value),
                "validation": _validation_contract(
                    strategy_id,
                    field.name,
                    value,
                    paths,
                    supported_entry_plan_modes=supported_entry_plan_modes or frozenset(),
                ),
                "hot_reload": False,
                "reload_behavior": RUNTIME_RESTART,
                "safety_critical": _is_safety_critical(field.name, paths),
            }
        )
    return entries


def _validation_contract(
    strategy_id: str,
    field_name: str,
    value: Any,
    paths: list[str],
    *,
    supported_entry_plan_modes: set[str] | frozenset[str],
) -> dict[str, Any]:
    key = f"{strategy_id}.{field_name}"
    validation: dict[str, Any] = {
        "accepted_value_shape": _value_type(value),
        "validated_by": f"{strategy_id}Config.from_mapping(...).__post_init__",
    }
    validation.update(_validation_overrides().get(key, {}))
    if isinstance(value, bool):
        validation.setdefault("allowed_values", [False, True])
    elif isinstance(value, Enum):
        validation.setdefault("allowed_values", [item.value for item in type(value)])
    elif isinstance(value, time):
        validation.setdefault("format", "HH:MM")
    elif isinstance(value, int) and not isinstance(value, bool):
        validation.setdefault("integer", True)
        if any(token in field_name for token in ("count", "size", "bars", "positions", "slots", "budget", "rank")):
            validation.setdefault("minimum", 0)
    elif isinstance(value, float):
        validation.setdefault("number", True)
        if field_name.endswith("_pct") or "_pct" in field_name:
            validation.setdefault("unit", "fraction_or_percent_as_documented_by_field")
    if any(path.endswith("entry.plan_mode") or path.endswith("entry.entry_plan_mode") for path in paths):
        validation["allowed_values"] = sorted(supported_entry_plan_modes)
    return validation


def _validation_overrides() -> dict[str, dict[str, Any]]:
    return {
        "KALCB.strategy_id": {"allowed_values": ["KALCB"], "fixed": True},
        "KALCB.timeframe": {"allowed_values": ["5m"], "fixed": True},
        "KALCB.execution_timeframe": {"allowed_values": ["5m_next_open"], "fixed": True},
        "KALCB.live_parity_fill_timing": {"allowed_values": ["next_5m_open"], "fixed": True},
        "KALCB.auction_mode": {"allowed_values": ["non_auction_continuous"], "fixed": True},
        "KALCB.opening_range_bars": {"allowed_values": [6], "fixed": True},
        "KALCB.ws_budget": {"minimum": 1, "maximum_binding": "ws_budget * ws_hot_regs_per_symbol <= ws_max_registrations - ws_reserved_execution_regs"},
        "KALCB.ws_max_registrations": {"minimum_binding": "must exceed ws_reserved_execution_regs"},
        "KALCB.frontier_size": {"minimum_binding": "must be at least ws_budget when frontier_enabled"},
        "KALCB.frontier_rotation_slots": {"minimum": 0, "maximum_binding": "must be <= ws_budget"},
        "KALCB.research_min_history_days": {"minimum": 20},
        "KALCB.research_min_accumulation_score": {"minimum": -1.0, "maximum": 1.0},
        "KALCB.research_min_structural_campaign_score": {"minimum": 0.0, "maximum": 10.0},
        "KALCB.partial_fraction": {"exclusive_minimum": 0.0, "exclusive_maximum": 1.0, "when": "use_partial_takes is true"},
        "KALCB.entry_plan_min_reclaim_closes": {"minimum": 1},
        "OLR.strategy_id": {"allowed_values": ["OLR"], "fixed": True},
        "OLR.timeframe": {"allowed_values": ["5m"], "fixed": True},
        "OLR.complete_universe_size": {"minimum": 1, "approved_deployment_value": 103},
        "OLR.research_top_long_count": {"minimum": 1},
        "OLR.afternoon_top_n": {"minimum": 1},
        "OLR.overnight_slot_count": {"minimum": 1},
        "OLR.premarket_frontier_size": {"minimum": 1},
        "OLR.research_min_history_days": {"minimum": 20},
        "OLR.afternoon_min_bar_count": {"minimum": 1},
        "OLR.afternoon_score_calibration_mode": {"allowed_values": ["raw", "exhaustion_adjusted"]},
        "OLR.entry_mode": {"allowed_values": ["close_auction", "decision_next_open"]},
        "OLR.exit_mode": {"allowed_values": ["next_close"], "fixed": True},
        "OLR.target_gross_exposure": {"minimum": 0.0, "maximum": 2.0},
        "OLR.max_position_pct": {"exclusive_minimum": 0.0, "maximum": 1.0},
        "OLR.min_selected": {"minimum": 1},
        "OLR.auction_nonfill_rate": {"minimum": 0.0, "maximum": 1.0},
    }


def _source_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "kalcb_config": KALCB_CONFIG_PATH,
        "kalcb_config_module": KALCB_CONFIG_MODULE_PATH,
        "olr_config_module": OLR_CONFIG_MODULE_PATH,
        "sector_map": SECTOR_MAP_PATH,
        "deployment_universe": UNIVERSE_PATH,
        "portfolio_policy": PORTFOLIO_POLICY_PATH,
        "oms_config": OMS_CONFIG_PATH,
    }
    return {
        name: {
            "path": path.as_posix(),
            "sha256": file_sha256(root / path),
            "required": True,
        }
        for name, path in paths.items()
    }


def _sector_map_contract(sector_map: Mapping[str, str]) -> dict[str, Any]:
    sectors = sorted(set(str(value) for value in sector_map.values()))
    return {
        "source_file": SECTOR_MAP_PATH.as_posix(),
        "editable_paths": ["sector_map.<symbol>", "<symbol>"],
        "symbol_count": len(sector_map),
        "sector_count": len(sectors),
        "sector_hash": canonical_json_hash(dict(sector_map)),
        "valid_range": {
            "symbol": "six-digit KRX ticker string",
            "sector": "non-empty canonical OMS sector label",
        },
        "hot_reload": False,
        "reload_behavior": RUNTIME_RESTART,
        "safety_critical": True,
    }


def _universe_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    symbols = tuple(str(item).zfill(6) for item in payload.get("symbols") or ())
    return {
        "source_file": UNIVERSE_PATH.as_posix(),
        "editable_paths": ["symbols", "symbol_count", "symbols_sha256", "complete_universe_size"],
        "schema_version": payload.get("schema_version", ""),
        "strategy": payload.get("strategy", "OLR"),
        "complete_universe_size": payload.get("complete_universe_size"),
        "symbol_count": payload.get("symbol_count", len(symbols)),
        "symbols_sha256": payload.get("symbols_sha256", ""),
        "computed_symbols_sha256": _symbol_list_sha256(symbols),
        "valid_range": {
            "symbols": "unique six-digit KRX ticker strings",
            "symbol_count": "must equal len(symbols)",
            "symbols_sha256": "must equal sha256 of newline-joined symbols with trailing newline",
        },
        "hot_reload": False,
        "reload_behavior": RUNTIME_RESTART,
        "safety_critical": True,
    }


def _portfolio_policy_contract(config: PortfolioPolicyConfig) -> dict[str, Any]:
    return {
        "source_file": PORTFOLIO_POLICY_PATH.as_posix(),
        "effective_policy": _jsonable(asdict(config)),
        "editable_parameters": [
            {
                "parameter_id": f"portfolio_policy.{field.name}",
                "source_files": [PORTFOLIO_POLICY_PATH.as_posix(), "deployment/olr_kalcb/portfolio.py"],
                "editable_paths": [field.name],
                "default_value": _jsonable(_field_default(field)),
                "effective_value": _jsonable(getattr(config, field.name)),
                "value_type": _value_type(getattr(config, field.name)),
                "validation": _portfolio_validation(field.name),
                "hot_reload": False,
                "reload_behavior": RUNTIME_RESTART,
                "safety_critical": True,
            }
            for field in fields(PortfolioPolicyConfig)
        ],
    }


def _oms_policy_contract(oms_config: Mapping[str, Any], risk_config: Mapping[str, Any]) -> dict[str, Any]:
    risk_paths = [f"risk.{field}" for field in RISK_CONFIG_FIELDS if field != "strategy_budgets"]
    strategy_budget_paths = [
        f"strategy_budgets.{strategy_id}.{key}"
        for strategy_id, budget in sorted(dict(oms_config.get("strategy_budgets") or {}).items())
        for key in sorted(dict(budget or {}))
    ]
    return {
        "source_file": OMS_CONFIG_PATH.as_posix(),
        "active_strategies": list(oms_config.get("active_strategies") or ()),
        "sector_map_path": oms_config.get("sector_map_path", ""),
        "effective_risk_config": _jsonable(risk_config),
        "editable_paths": sorted([*risk_paths, *strategy_budget_paths, "active_strategies", "sector_map_path"]),
        "validation": {
            "validated_by": "oms.config_loader.build_risk_config and oms.risk.RiskGateway",
            "unknown_sector_policy": ["allow", "block"],
            "default_stop_protection_mode": ["oms_watcher"],
            "stop_exit_order_type": ["MARKET"],
            "percentage_fields": "fractions in [0, 1] unless a strategy-specific test documents otherwise",
        },
        "hot_reload": False,
        "reload_behavior": OMS_RESTART,
        "safety_critical": True,
    }


def _event_stream_contract() -> dict[str, dict[str, Any]]:
    streams = {}
    for event_type, directory in sorted(EVENT_DIRS.items()):
        streams[event_type] = {
            "event_type": event_type,
            "directory": directory,
            "schema_version": event_schema_version(event_type),
            "scope": event_scope(event_type),
            "priority": event_priority(event_type),
            "payload_identity_required": event_type
            in {
                "trade",
                "order",
                "fill",
                "portfolio_rule",
                "risk_decision",
                "oms_intent",
                "deployment",
                "config_snapshot",
                "resource_plan",
                "session_closeout",
            },
        }
    return streams


def _portfolio_validation(field_name: str) -> dict[str, Any]:
    if field_name == "strategy_priority":
        return {
            "accepted_value_shape": "array[string]",
            "allowed_values": ["KALCB", "OLR"],
            "rule": "priority order controls same-timestamp cross-strategy arbitration",
        }
    return {
        "accepted_value_shape": "number",
        "minimum": 0.0,
        "unit": "KRW notional",
    }


def _portable_policy_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(payload.get("portfolio_policy") or payload)
    if "strategy_priority" in raw:
        raw["strategy_priority"] = tuple(raw["strategy_priority"] or ())
    return raw


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _effective_oms_risk_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config or {})
    risk_section = payload if _looks_effective_risk_config(payload) else dict(payload.get("risk") or {})
    result = {
        field: risk_section.get(field, RISK_CONFIG_DEFAULTS[field])
        for field in RISK_CONFIG_FIELDS
        if field not in {"strategy_budgets", "regime_exposure_caps", "current_regime"}
    }
    result["strategy_budgets"] = payload.get("strategy_budgets", RISK_CONFIG_DEFAULTS["strategy_budgets"])
    result["regime_exposure_caps"] = payload.get("regime_exposure_caps", RISK_CONFIG_DEFAULTS["regime_exposure_caps"])
    result["current_regime"] = payload.get("current_regime", RISK_CONFIG_DEFAULTS["current_regime"])
    policy = str(result["unknown_sector_policy"] or "allow").lower().strip()
    result["unknown_sector_policy"] = "block" if policy in {"block", "reject"} else "allow"
    return result


def _looks_effective_risk_config(payload: Mapping[str, Any]) -> bool:
    return any(field in payload for field in RISK_CONFIG_FIELDS if field not in {"strategy_budgets", "regime_exposure_caps"})


def _field_default(field: Any) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return None


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, time):
        return "time"
    if isinstance(value, Enum):
        return "enum"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple, set)):
        return "array"
    if value is None:
        return "null"
    return "string"


def _is_safety_critical(field_name: str, paths: list[str]) -> bool:
    haystack = " ".join([field_name, *paths]).lower()
    return any(keyword in haystack for keyword in SAFETY_KEYWORDS)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return dict(payload)


def _load_sector_map(path: Path) -> dict[str, str]:
    payload = _load_yaml(path)
    raw = payload.get("sector_map", payload)
    if not isinstance(raw, Mapping):
        return {}
    return {str(key).zfill(6): str(value).upper().strip() for key, value in raw.items()}


def _symbol_list_sha256(symbols: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(symbols) + "\n").encode("utf-8")).hexdigest()


def _contract_hash_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(contract)
    payload.pop("contract_hash", None)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return value.as_posix()
    return value
