"""OMS config loading and effective risk-config normalization."""

from __future__ import annotations

import os
import json
import hashlib
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from strategy_common.sector_map import normalize_sector_map

from .risk import RiskConfig


RISK_CONFIG_FIELDS = set(RiskConfig.__dataclass_fields__)
DEFAULT_ACTIVE_STRATEGIES = ("KALCB", "OLR")


def oms_config_search_paths(config_path: str | Path | None = None) -> tuple[Path, ...]:
    raw_paths = (
        config_path,
        os.environ.get("OMS_CONFIG_PATH"),
        "config/oms_config.yaml",
        "../config/oms_config.yaml",
        Path(__file__).resolve().parent.parent / "config" / "oms_config.yaml",
    )
    return tuple(Path(path) for path in raw_paths if path not in (None, ""))


def load_oms_config_with_source(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    for path in oms_config_search_paths(config_path):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a YAML mapping")
        return dict(payload), path.resolve()
    return {}, None


def load_oms_config(config_path: str | Path | None = None) -> dict[str, Any]:
    payload, _source = load_oms_config_with_source(config_path)
    return payload


def build_risk_config(config: Mapping[str, Any] | None = None) -> RiskConfig:
    payload = dict(config or {})
    if _looks_effective_risk_config(payload):
        return RiskConfig(
            daily_loss_warn_pct=payload.get("daily_loss_warn_pct", 0.02),
            daily_loss_halt_pct=payload.get("daily_loss_halt_pct", 0.03),
            max_gross_exposure_pct=payload.get("max_gross_exposure_pct", 0.80),
            max_net_exposure_pct=payload.get("max_net_exposure_pct", 0.60),
            max_position_pct=payload.get("max_position_pct", 0.15),
            max_positions_count=payload.get("max_positions_count", 10),
            max_sector_pct=payload.get("max_sector_pct", 0.30),
            unknown_sector_policy=payload.get("unknown_sector_policy", "allow"),
            strategy_budgets=payload.get("strategy_budgets"),
            max_spread_bps=payload.get("max_spread_bps", 50.0),
            vi_cooldown_sec=payload.get("vi_cooldown_sec", 600.0),
            regime_exposure_caps=payload.get("regime_exposure_caps"),
            current_regime=payload.get("current_regime", "NORMAL"),
            require_durable_stops=payload.get("require_durable_stops", True),
            default_stop_protection_mode=payload.get("default_stop_protection_mode", "oms_watcher"),
            allow_synthetic_stop_only=payload.get("allow_synthetic_stop_only", False),
            stop_price_stale_after_sec=payload.get("stop_price_stale_after_sec", 30.0),
            stop_watcher_interval_sec=payload.get("stop_watcher_interval_sec", 5.0),
            stop_exit_order_type=payload.get("stop_exit_order_type", "MARKET"),
            stop_protection_emergency_override=payload.get("stop_protection_emergency_override", False),
        )

    risk_section = dict(payload.get("risk") or {})
    return RiskConfig(
        daily_loss_warn_pct=risk_section.get("daily_loss_warn_pct", 0.02),
        daily_loss_halt_pct=risk_section.get("daily_loss_halt_pct", 0.03),
        max_gross_exposure_pct=risk_section.get("max_gross_exposure_pct", 0.80),
        max_net_exposure_pct=risk_section.get("max_net_exposure_pct", 0.60),
        max_position_pct=risk_section.get("max_position_pct", 0.15),
        max_positions_count=risk_section.get("max_positions_count", 10),
        max_sector_pct=risk_section.get("max_sector_pct", 0.30),
        unknown_sector_policy=risk_section.get("unknown_sector_policy", payload.get("unknown_sector_policy", "allow")),
        strategy_budgets=payload.get("strategy_budgets"),
        max_spread_bps=risk_section.get("max_spread_bps", 50.0),
        vi_cooldown_sec=risk_section.get("vi_cooldown_sec", 600.0),
        regime_exposure_caps=payload.get("regime_exposure_caps") or None,
        current_regime=payload.get("current_regime", "NORMAL"),
        require_durable_stops=risk_section.get("require_durable_stops", True),
        default_stop_protection_mode=risk_section.get("default_stop_protection_mode", "oms_watcher"),
        allow_synthetic_stop_only=risk_section.get("allow_synthetic_stop_only", False),
        stop_price_stale_after_sec=risk_section.get("stop_price_stale_after_sec", 30.0),
        stop_watcher_interval_sec=risk_section.get("stop_watcher_interval_sec", 5.0),
        stop_exit_order_type=risk_section.get("stop_exit_order_type", "MARKET"),
        stop_protection_emergency_override=risk_section.get("stop_protection_emergency_override", False),
    )


def effective_risk_config_payload(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return asdict(build_risk_config(config or {}))


def load_effective_risk_config_payload(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    config, source = load_oms_config_with_source(config_path)
    return effective_risk_config_payload(config), source


def configured_active_strategy_ids(config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    payload = dict(config or {})
    raw = payload.get("active_strategies")
    if raw is None:
        registry = payload.get("strategy_registry")
        if isinstance(registry, Mapping):
            raw = registry.get("strategy_ids") or registry.get("active_strategy_ids")
    if raw is None:
        raw = DEFAULT_ACTIVE_STRATEGIES
    return tuple(
        dict.fromkeys(
            str(item).upper().strip()
            for item in _as_sequence(raw)
            if str(item).strip()
        )
    )


def missing_strategy_budgets(
    risk_config: RiskConfig,
    active_strategy_ids: Sequence[str],
) -> tuple[str, ...]:
    budgets = {str(key).upper().strip() for key in dict(risk_config.strategy_budgets or {})}
    return tuple(sid for sid in active_strategy_ids if sid and sid not in budgets)


def load_oms_sector_map(
    config: Mapping[str, Any] | None = None,
    *,
    config_source: str | Path | None = None,
    default_path: str | Path = "config/olr/sector_map.yaml",
) -> tuple[dict[str, str], Path | None]:
    payload = dict(config or {})
    risk_section = dict(payload.get("risk") or {})
    raw_map = payload.get("sector_map", risk_section.get("sector_map"))
    if isinstance(raw_map, Mapping):
        return normalize_sector_map(raw_map), None

    raw_path = (
        raw_map
        if isinstance(raw_map, str)
        else payload.get("sector_map_path", risk_section.get("sector_map_path", default_path))
    )
    path = _resolve_config_relative_path(raw_path, config_source=config_source)
    if path is None or not path.exists():
        return {}, path
    loaded = _load_sector_map_from_yaml(path)
    return normalize_sector_map(loaded), path.resolve()


def stable_mapping_hash(payload: Mapping[str, Any] | None) -> str:
    raw = json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _looks_effective_risk_config(payload: Mapping[str, Any]) -> bool:
    return any(field in payload for field in RISK_CONFIG_FIELDS if field not in {"strategy_budgets", "regime_exposure_caps"})


def _as_sequence(raw: Any) -> tuple[Any, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    if isinstance(raw, Sequence):
        return tuple(raw)
    return (raw,)


def _resolve_config_relative_path(raw: Any, *, config_source: str | Path | None) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidates = []
    if config_source is not None:
        candidates.append(Path(config_source).resolve().parent / path)
    candidates.extend((Path.cwd() / path, Path.cwd() / "config" / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _load_sector_map_from_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("sector_map", payload)
    return dict(raw or {}) if isinstance(raw, Mapping) else {}
