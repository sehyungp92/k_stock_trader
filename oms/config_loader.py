"""OMS config loading and effective risk-config normalization."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from .risk import RiskConfig


RISK_CONFIG_FIELDS = set(RiskConfig.__dataclass_fields__)


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
            strategy_budgets=payload.get("strategy_budgets"),
            max_spread_bps=payload.get("max_spread_bps", 50.0),
            vi_cooldown_sec=payload.get("vi_cooldown_sec", 600.0),
            regime_exposure_caps=payload.get("regime_exposure_caps"),
            current_regime=payload.get("current_regime", "NORMAL"),
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
        strategy_budgets=payload.get("strategy_budgets"),
        max_spread_bps=risk_section.get("max_spread_bps", 50.0),
        vi_cooldown_sec=risk_section.get("vi_cooldown_sec", 600.0),
        regime_exposure_caps=payload.get("regime_exposure_caps") or None,
        current_regime=payload.get("current_regime", "NORMAL"),
    )


def effective_risk_config_payload(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return asdict(build_risk_config(config or {}))


def load_effective_risk_config_payload(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    config, source = load_oms_config_with_source(config_path)
    return effective_risk_config_payload(config), source


def _looks_effective_risk_config(payload: Mapping[str, Any]) -> bool:
    return any(field in payload for field in RISK_CONFIG_FIELDS if field not in {"strategy_budgets", "regime_exposure_caps"})
