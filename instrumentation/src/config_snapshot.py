"""Effective config hashing and redaction helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lineage import LineageContext, deployment_id_for, redact_mapping, stable_hash


CONFIG_VERSION_FIELDS = (
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "kis_resource_plan_hash",
)


def file_hash(path: str | Path) -> str:
    import hashlib

    target = Path(path)
    if not target.is_file():
        return ""
    h = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def effective_config_snapshot(
    *,
    strategy_configs: Mapping[str, Any] | None = None,
    portfolio_config: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
    allocation_state: Mapping[str, Any] | None = None,
    strategy_registry: Mapping[str, Any] | None = None,
    resource_plan: Mapping[str, Any] | None = None,
    source_files: Sequence[str | Path] = (),
    environment: Mapping[str, Any] | None = None,
    lineage: LineageContext | None = None,
) -> dict[str, Any]:
    strategies_redacted, strategy_redacted_keys = redact_mapping(dict(strategy_configs or {}))
    portfolio_redacted, portfolio_redacted_keys = redact_mapping(dict(portfolio_config or {}))
    risk_redacted, risk_redacted_keys = redact_mapping(dict(risk_config or {}))
    allocation_redacted, allocation_redacted_keys = redact_mapping(dict(allocation_state or {}))
    registry_redacted, registry_redacted_keys = redact_mapping(dict(strategy_registry or {}))
    resource_redacted, resource_redacted_keys = redact_mapping(dict(resource_plan or {}))
    env_redacted, env_redacted_keys = redact_mapping(dict(environment or {}))
    computed_versions = {
        "config_version": stable_hash(strategies_redacted),
        "portfolio_config_version": stable_hash(portfolio_redacted),
        "risk_config_version": stable_hash(risk_redacted),
        "allocation_version": stable_hash(allocation_redacted),
        "strategy_registry_version": stable_hash(registry_redacted),
        "kis_resource_plan_hash": str(resource_redacted.get("plan_hash") or stable_hash(resource_redacted)) if resource_redacted else "",
    }
    versions = dict(computed_versions)
    if lineage is not None:
        for field in CONFIG_VERSION_FIELDS:
            value = str(getattr(lineage, field, "") or "")
            if value:
                versions[field] = value
    active_strategy_budget_status = _active_strategy_budget_status(registry_redacted, risk_redacted)
    source_rows = [
        {"path": str(path), "sha256": file_hash(path)}
        for path in source_files
    ]
    deployment_id = (
        lineage.deployment_id
        if lineage is not None and lineage.deployment_id
        else deployment_id_for({"versions": versions, "sources": source_rows})
    )
    snapshot = {
        "record_type": "config_snapshot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deployment_id": deployment_id,
        **versions,
        "effective_configs": {
            "strategies": strategies_redacted,
            "portfolio": portfolio_redacted,
            "risk": risk_redacted,
            "allocation": allocation_redacted,
            "strategy_registry": registry_redacted,
            "kis_resource_plan": resource_redacted,
        },
        "source_files": source_rows,
        "redacted_environment": env_redacted,
        "redacted_keys": sorted(
            dict.fromkeys(
                [
                    *_prefix_keys("strategies", strategy_redacted_keys),
                    *_prefix_keys("portfolio", portfolio_redacted_keys),
                    *_prefix_keys("risk", risk_redacted_keys),
                    *_prefix_keys("allocation", allocation_redacted_keys),
                    *_prefix_keys("strategy_registry", registry_redacted_keys),
                    *_prefix_keys("kis_resource_plan", resource_redacted_keys),
                    *_prefix_keys("environment", env_redacted_keys),
                ]
            )
        ),
        "active_strategy_budget_status": active_strategy_budget_status,
    }
    if computed_versions != versions:
        snapshot["computed_versions"] = computed_versions
    return snapshot


def snapshot_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def _active_strategy_budget_status(registry: Mapping[str, Any], risk: Mapping[str, Any]) -> dict[str, str]:
    raw_ids = registry.get("strategy_ids") or registry.get("active_strategy_ids") or ()
    strategy_ids = [str(item).upper().strip() for item in raw_ids if str(item).strip()]
    budgets = dict(risk.get("strategy_budgets") or {})
    result: dict[str, str] = {}
    for sid in strategy_ids:
        result[sid] = "configured" if sid in budgets else "missing_uses_global_limits"
    return result


def _prefix_keys(prefix: str, keys: list[str]) -> list[str]:
    return [f"{prefix}.{key}" for key in keys]
