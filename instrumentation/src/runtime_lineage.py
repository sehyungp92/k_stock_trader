"""Shared runtime deployment/version lineage helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lineage import LineageContext, deployment_id_for, stable_hash


RUNTIME_DEPLOYMENT_LINEAGE_FILENAME = "runtime_deployment_lineage.json"
RUNTIME_DEPLOYMENT_LINEAGE_ENV = "RUNTIME_DEPLOYMENT_LINEAGE_PATH"


VERSION_FIELDS = (
    "strategy_version",
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "kis_resource_plan_hash",
)


def runtime_strategy_ids(payload: Mapping[str, Any], strategy_ids: Sequence[str] | None = None) -> tuple[str, ...]:
    raw = strategy_ids if strategy_ids is not None else payload.get("strategy_ids") or payload.get("active_strategy_ids") or ()
    return tuple(sid for sid in (str(item).upper().strip() for item in raw) if sid)


def runtime_versions_from_manifest(
    payload: Mapping[str, Any],
    *,
    strategy_ids: Sequence[str] | None = None,
    risk_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    manifest = dict(payload or {})
    ids = runtime_strategy_ids(manifest, strategy_ids)
    risk_payload = dict(risk_config or runtime_risk_config_from_manifest(manifest))
    return {
        "strategy_version": runtime_strategy_version(manifest, ids),
        "config_version": stable_hash(manifest.get("strategy_configs") or {}),
        "portfolio_config_version": stable_hash(manifest.get("portfolio_policy_config") or {}),
        "risk_config_version": stable_hash(risk_payload),
        "allocation_version": stable_hash(runtime_allocations_from_positions(manifest.get("initial_positions"))),
        "strategy_registry_version": stable_hash(
            {
                "strategy_ids": ids,
                "mode": manifest.get("mode"),
                "staged_artifacts": runtime_artifact_version_rows(manifest, ids),
            }
        ),
        "kis_resource_plan_hash": str(manifest.get("kis_resource_plan_hash") or ""),
    }


def runtime_strategy_version(payload: Mapping[str, Any], strategy_ids: Sequence[str] | None = None) -> str:
    ids = runtime_strategy_ids(payload, strategy_ids)
    return f"olr_kalcb:{stable_hash({'strategy_ids': list(ids), 'artifacts': runtime_artifact_version_rows(payload, ids)})}"


def runtime_artifact_version_rows(payload: Mapping[str, Any], strategy_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in payload.get("staged_artifacts") or ():
        row = dict(item or {})
        strategy_id = str(row.get("strategy_id") or "").upper().strip()
        if not strategy_id:
            continue
        rows.append(
            {
                "strategy_id": strategy_id,
                "artifact_hash": str(row.get("artifact_hash") or ""),
                "artifact_stage": str(row.get("artifact_stage") or row.get("stage") or ""),
                "source_fingerprint": str(row.get("source_fingerprint") or ""),
            }
        )
    if rows:
        return sorted(rows, key=lambda row: row["strategy_id"])

    artifacts = payload.get("strategy_artifacts")
    if isinstance(artifacts, Mapping):
        for sid, raw in sorted(dict(artifacts).items(), key=lambda item: str(item[0])):
            item = _artifact_payload(raw)
            strategy_id = str(sid).upper().strip()
            if not strategy_id:
                continue
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "artifact_hash": str(item.get("artifact_hash") or ""),
                    "artifact_stage": str(item.get("artifact_stage") or item.get("stage") or ""),
                    "source_fingerprint": str(item.get("source_fingerprint") or ""),
                }
            )
        if rows:
            return rows

    configs = dict(payload.get("strategy_configs") or {})
    ids = runtime_strategy_ids(payload, strategy_ids)
    return [
        {"strategy_id": strategy_id, "config_hash": stable_hash(configs.get(strategy_id) or {})}
        for strategy_id in sorted(ids)
    ]


def runtime_risk_config_from_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        payload.get("risk_config")
        or payload.get("risk_config_effective")
        or payload.get("oms_risk_config")
        or payload.get("oms_config")
    )
    if isinstance(raw, Mapping) and raw:
        return _effective_risk_config_payload(raw)
    return _load_runtime_risk_config_payload()


def runtime_deployment_id(versions: Mapping[str, Any], *, code_sha: str = "") -> str:
    return deployment_id_for({"manifest": {key: versions.get(key) for key in VERSION_FIELDS}, "code_sha": str(code_sha or "")})


def runtime_lineage_payload(
    manifest: Mapping[str, Any],
    *,
    lineage: LineageContext | None = None,
    strategy_ids: Sequence[str] | None = None,
    risk_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    versions = runtime_versions_from_manifest(manifest, strategy_ids=strategy_ids, risk_config=risk_config)
    code_sha = str(getattr(lineage, "code_sha", "") or "")
    deployment_id = str(getattr(lineage, "deployment_id", "") or "") or runtime_deployment_id(versions, code_sha=code_sha)
    return {
        "deployment_id": deployment_id,
        "code_sha": code_sha,
        "strategy_ids": list(runtime_strategy_ids(manifest, strategy_ids)),
        **versions,
    }


def write_runtime_deployment_lineage(data_dir: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(data_dir) / RUNTIME_DEPLOYMENT_LINEAGE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_runtime_deployment_lineage(data_dir: str | Path | None = None, *, explicit_path: str | Path | None = None) -> dict[str, Any]:
    for path in _runtime_lineage_paths(data_dir, explicit_path=explicit_path):
        try:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            return {
                key: value
                for key, value in dict(payload).items()
                if key
                in {
                    *VERSION_FIELDS,
                    "deployment_id",
                    "code_sha",
                    "portfolio_id",
                    "account_alias",
                    "portfolio_policy_hash",
                }
                and value not in (None, "")
            }
    return {}


def runtime_allocations_from_positions(raw: Any) -> list[dict[str, Any]]:
    positions = _positions_from_manifest(raw)
    rows: list[dict[str, Any]] = []
    for position in positions:
        symbol = str(position.get("symbol") or "").zfill(6)
        for sid, allocation in sorted(dict(position.get("allocations") or {}).items()):
            item = dict(allocation or {}) if isinstance(allocation, Mapping) else {"qty": allocation}
            rows.append({"symbol": symbol, "strategy_id": str(sid).upper().strip(), **item})
    return rows


def _runtime_lineage_paths(data_dir: str | Path | None, *, explicit_path: str | Path | None) -> tuple[Path, ...]:
    raw = (
        explicit_path,
        os.environ.get(RUNTIME_DEPLOYMENT_LINEAGE_ENV),
        Path(data_dir) / RUNTIME_DEPLOYMENT_LINEAGE_FILENAME if data_dir not in (None, "") else None,
    )
    return tuple(Path(path) for path in raw if path not in (None, ""))


def _effective_risk_config_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from oms.config_loader import effective_risk_config_payload

        return effective_risk_config_payload(raw)
    except Exception:
        return dict(raw)


def _load_runtime_risk_config_payload() -> dict[str, Any]:
    try:
        from oms.config_loader import load_effective_risk_config_payload

        payload, _source = load_effective_risk_config_payload()
        return payload
    except Exception:
        try:
            from oms.risk import RiskConfig

            config = RiskConfig()
            if is_dataclass(config):
                return asdict(config)
            return dict(getattr(config, "__dict__", {}) or {})
        except Exception:
            return {}


def _artifact_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    metadata = dict(getattr(raw, "metadata", {}) or {})
    return {
        "artifact_hash": getattr(raw, "artifact_hash", ""),
        "artifact_stage": metadata.get("artifact_stage", ""),
        "source_fingerprint": getattr(raw, "source_fingerprint", ""),
    }


def _positions_from_manifest(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [
            {"symbol": str(symbol).zfill(6), **(dict(row) if isinstance(row, Mapping) else {"value": row})}
            for symbol, row in sorted(dict(raw).items(), key=lambda item: str(item[0]))
        ]
    if isinstance(raw, (list, tuple)):
        result: list[dict[str, Any]] = []
        for row in raw:
            item = dict(row or {}) if isinstance(row, Mapping) else {"value": row}
            item["symbol"] = str(item.get("symbol") or "").zfill(6)
            result.append(item)
        return result
    return []
