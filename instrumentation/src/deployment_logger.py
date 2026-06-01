"""Deployment and config snapshot emitters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config_snapshot import effective_config_snapshot
from .event_writer import JSONLEventWriter
from .lineage import LineageContext, context_from_env, deployment_id_for


class DeploymentLogger:
    def __init__(self, data_dir: str | Path = "instrumentation/data", *, lineage: LineageContext | None = None) -> None:
        self.lineage = lineage or context_from_env(data_source_id="runtime_session")
        self.writer = JSONLEventWriter(data_dir, lineage=self.lineage)

    def emit_deployment(
        self,
        *,
        status: str,
        mode: str,
        strategy_ids: list[str] | tuple[str, ...],
        source: str,
        payload: Mapping[str, Any] | None = None,
        lineage: LineageContext | None = None,
    ) -> dict[str, Any] | None:
        ctx = lineage or self.lineage
        base = dict(payload or {})
        deployment_id = ctx.deployment_id or str(base.get("deployment_id") or "") or deployment_id_for(
            {
                "mode": mode,
                "strategy_ids": list(strategy_ids),
                "code_sha": ctx.code_sha,
                "config_version": ctx.config_version,
                "portfolio_config_version": ctx.portfolio_config_version,
                "risk_config_version": ctx.risk_config_version,
                "allocation_version": ctx.allocation_version,
                "kis_resource_plan_hash": ctx.kis_resource_plan_hash,
            }
        )
        ctx = ctx.with_overrides(deployment_id=deployment_id)
        event = {
            "record_type": "deployment",
            "deployment_id": deployment_id,
            "mode": mode,
            "strategy_ids": [str(item).upper().strip() for item in strategy_ids],
            "status": status,
            "source": source,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            **base,
        }
        return self.writer.write("deployment", event, payload_key=f"{deployment_id}:{status}:{source}", lineage=ctx, scope="portfolio")

    def emit_config_snapshot(
        self,
        *,
        strategy_configs: Mapping[str, Any] | None = None,
        portfolio_config: Mapping[str, Any] | None = None,
        risk_config: Mapping[str, Any] | None = None,
        allocation_state: Mapping[str, Any] | None = None,
        strategy_registry: Mapping[str, Any] | None = None,
        resource_plan: Mapping[str, Any] | None = None,
        source_files: list[str | Path] | tuple[str | Path, ...] = (),
        environment: Mapping[str, Any] | None = None,
        lineage: LineageContext | None = None,
    ) -> dict[str, Any] | None:
        ctx = lineage or self.lineage
        snapshot = effective_config_snapshot(
            strategy_configs=strategy_configs,
            portfolio_config=portfolio_config,
            risk_config=risk_config,
            allocation_state=allocation_state,
            strategy_registry=strategy_registry,
            resource_plan=resource_plan,
            source_files=source_files,
            environment=environment,
            lineage=ctx,
        )
        ctx = ctx.with_overrides(
            deployment_id=snapshot.get("deployment_id"),
            config_version=snapshot.get("config_version"),
            portfolio_config_version=snapshot.get("portfolio_config_version"),
            risk_config_version=snapshot.get("risk_config_version"),
            allocation_version=snapshot.get("allocation_version"),
            strategy_registry_version=snapshot.get("strategy_registry_version"),
            kis_resource_plan_hash=snapshot.get("kis_resource_plan_hash"),
        )
        return self.writer.write(
            "config_snapshot",
            snapshot,
            payload_key=str(snapshot.get("deployment_id") or snapshot.get("config_version") or ""),
            lineage=ctx,
            scope="portfolio",
        )
