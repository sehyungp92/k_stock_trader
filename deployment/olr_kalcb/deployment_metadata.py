from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from instrumentation.src.runtime_lineage import runtime_deployment_id, runtime_versions_from_manifest

from .hashing import canonical_json_hash, file_sha256


METADATA_SOURCE = "live_bot_runtime_deployment_metadata_v1"
TELEMETRY_SCHEMA_VERSION = "olr_kalcb_decision_stream_v1"
DEFAULT_CONTRACT_PATH = Path("trading_assistant_backtest/contracts/k_stock_olr_kalcb/strategy_plugin_contract.json")
ALLOWED_METADATA_SOURCES = {"live_bot_runtime_deployment_metadata_v1", "vps_live_bot_runtime_deployment_metadata_v1"}
ALLOWED_EMISSION_ENVIRONMENTS = {"live_bot", "vps", "paper_vps", "production_vps"}


class DeploymentMetadataError(RuntimeError):
    """Raised when approval-grade deployment metadata cannot be emitted."""


def emit_deployment_metadata(
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    contract_path: str | Path | None = None,
    mode: str,
    strategy_ids: tuple[str, ...] | list[str],
    strategy_configs: Mapping[str, Any] | None = None,
    portfolio_policy_config: Mapping[str, Any] | None = None,
    strategy_artifacts: Mapping[str, Any] | None = None,
    initial_positions: Any | None = None,
    kis_resource_plan_hash: str = "",
    deployment_id: str = "",
    runtime_started_at_utc: str | datetime | None = None,
    runtime_entrypoint: str = "",
    runtime_instance_id: str = "",
    emission_environment: str = "",
    metadata_source: str = METADATA_SOURCE,
) -> dict[str, Any]:
    """Build and write the approval-grade deployment metadata artifact.

    This is stricter than ordinary telemetry. If any approval-critical
    provenance is missing or ambiguous, the function raises and writes nothing.
    """

    root = Path(repo_root or Path.cwd()).resolve()
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    metadata = build_deployment_metadata(
        repo_root=root,
        contract_path=contract_path,
        mode=mode,
        strategy_ids=tuple(strategy_ids),
        strategy_configs=strategy_configs,
        portfolio_policy_config=portfolio_policy_config,
        strategy_artifacts=strategy_artifacts,
        initial_positions=initial_positions,
        kis_resource_plan_hash=kis_resource_plan_hash,
        deployment_id=deployment_id,
        runtime_started_at_utc=runtime_started_at_utc,
        runtime_entrypoint=runtime_entrypoint,
        runtime_instance_id=runtime_instance_id,
        emission_environment=emission_environment,
        metadata_source=metadata_source,
        ignore_worktree_paths=(target,),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def build_deployment_metadata(
    *,
    repo_root: str | Path,
    contract_path: str | Path | None = None,
    mode: str,
    strategy_ids: tuple[str, ...] | list[str],
    strategy_configs: Mapping[str, Any] | None = None,
    portfolio_policy_config: Mapping[str, Any] | None = None,
    strategy_artifacts: Mapping[str, Any] | None = None,
    initial_positions: Any | None = None,
    kis_resource_plan_hash: str = "",
    deployment_id: str = "",
    runtime_started_at_utc: str | datetime | None = None,
    runtime_entrypoint: str = "",
    runtime_instance_id: str = "",
    emission_environment: str = "",
    metadata_source: str = METADATA_SOURCE,
    source_control: Mapping[str, Any] | None = None,
    ignore_worktree_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source = dict(source_control or _source_control(root, ignore_paths=ignore_worktree_paths))
    remote = _require_real_remote(str(source.get("repo_url") or ""))
    commit = _require_full_commit_sha(str(source.get("commit_sha") or ""))
    if not bool(source.get("worktree_clean")):
        raise DeploymentMetadataError("source control worktree must be clean for deployment metadata")

    metadata_source = str(metadata_source or METADATA_SOURCE)
    if metadata_source not in ALLOWED_METADATA_SOURCES:
        raise DeploymentMetadataError(f"unsupported metadata_source={metadata_source!r}")
    normalized_strategy_ids = tuple(
        sid for sid in (str(item).upper().strip() for item in strategy_ids) if sid
    )
    environment = _emission_environment(mode, emission_environment)
    contract = _resolve_contract_path(root, contract_path)
    contract_hash = file_sha256(contract)
    artifacts = _artifact_versions(strategy_artifacts)
    manifest = {
        "mode": mode,
        "strategy_ids": list(normalized_strategy_ids),
        "strategy_configs": strategy_configs or {},
        "portfolio_policy_config": portfolio_policy_config or {},
        "strategy_artifacts": strategy_artifacts or {},
        "initial_positions": initial_positions,
        "kis_resource_plan_hash": kis_resource_plan_hash,
    }
    versions = runtime_versions_from_manifest(manifest, strategy_ids=normalized_strategy_ids)
    config_hash = versions["config_version"]
    legacy_config_hash = canonical_json_hash(
        {
            "strategy_configs": strategy_configs or {},
            "portfolio_policy_config": portfolio_policy_config or {},
            "kis_resource_plan_hash": kis_resource_plan_hash,
        }
    )
    stable_deployment_id = deployment_id or runtime_deployment_id(versions, code_sha=commit[:12])
    runtime_started = _iso_utc(runtime_started_at_utc)
    instance_id = runtime_instance_id or f"runtime:{canonical_json_hash({'deployment_id': stable_deployment_id, 'started_at': runtime_started})[:16]}"
    contract_path_value = _display_contract_path(root, contract)

    return {
        "metadata_source": metadata_source,
        "emission_environment": environment,
        "repo_url": remote,
        "source_control_origin": remote,
        "deployed_commit_sha": commit,
        "source_control_commit_sha": commit,
        "source_control_worktree_clean": True,
        "bot_id": "k_stock_trader",
        "portfolio_id": "olr_kalcb",
        "strategy_id": "OLR_KALCB",
        "config_hash": config_hash,
        "legacy_config_hash": legacy_config_hash,
        "strategy_version": versions["strategy_version"],
        "config_version": versions["config_version"],
        "portfolio_config_version": versions["portfolio_config_version"],
        "risk_config_version": versions["risk_config_version"],
        "allocation_version": versions["allocation_version"],
        "strategy_registry_version": versions["strategy_registry_version"],
        "deployment_id": stable_deployment_id,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "strategy_plugin_contract_path": contract_path_value,
        "strategy_plugin_contract_hash": contract_hash,
        "emitted_at_utc": _iso_utc(None),
        "live_runtime_started_at_utc": runtime_started,
        "runtime_entrypoint": runtime_entrypoint or "deployment.olr_kalcb.runtime:prepare_runtime_session",
        "runtime_instance_id": instance_id,
        "runtime_host_fingerprint": _host_fingerprint(),
        "dry_run": str(mode or "").lower().strip() == "dry_run",
        "strategy_ids": list(normalized_strategy_ids),
        "strategy_artifacts": artifacts,
        "kis_resource_plan_hash": str(kis_resource_plan_hash or ""),
    }


def _source_control(repo_root: Path, *, ignore_paths: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    return {
        "repo_url": _git(repo_root, "remote", "get-url", "origin"),
        "commit_sha": _git(repo_root, "rev-parse", "HEAD"),
        "worktree_clean": _git_status(repo_root, ignore_paths=ignore_paths) == "",
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5,
        )
    except Exception as exc:
        raise DeploymentMetadataError(f"git {' '.join(args)} failed") from exc
    return result.stdout.strip()


def _git_status(repo_root: Path, *, ignore_paths: tuple[str | Path, ...] = ()) -> str:
    args = ["status", "--porcelain", "--untracked-files=all"]
    excludes = tuple(_relative_git_path(repo_root, path) for path in ignore_paths)
    excludes = tuple(path for path in excludes if path)
    if excludes:
        args.extend(["--", ".", *(f":(exclude){path}" for path in excludes)])
    return _git(repo_root, *args)


def _relative_git_path(repo_root: Path, path: str | Path) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = repo_root / target
    try:
        return target.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return ""


def _require_real_remote(remote: str) -> str:
    value = str(remote or "").strip()
    lower = value.lower()
    if not value or lower.startswith(("local://", "file://")):
        raise DeploymentMetadataError("repo_url/source_control_origin must be a real remote")
    if "://" in value:
        scheme = lower.split("://", 1)[0]
        if scheme in {"http", "https", "ssh", "git"}:
            return value
    if value.startswith("git@") and ":" in value:
        return value
    raise DeploymentMetadataError("repo_url/source_control_origin must be a real remote")


def _require_full_commit_sha(commit: str) -> str:
    value = str(commit or "").strip()
    if len(value) not in {40, 64} or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise DeploymentMetadataError("source control commit sha must be a full git object id")
    return value


def _emission_environment(mode: str, value: str) -> str:
    if value:
        environment = str(value).strip()
    elif str(mode or "").lower().strip() == "live":
        environment = "production_vps"
    else:
        environment = "paper_vps"
    if environment not in ALLOWED_EMISSION_ENVIRONMENTS:
        raise DeploymentMetadataError(f"unsupported emission_environment={environment!r}")
    return environment


def _resolve_contract_path(repo_root: Path, contract_path: str | Path | None) -> Path:
    raw = Path(contract_path or os.environ.get("OLR_KALCB_STRATEGY_PLUGIN_CONTRACT", "") or DEFAULT_CONTRACT_PATH)
    contract = raw if raw.is_absolute() else repo_root / raw
    if not contract.is_file():
        raise DeploymentMetadataError(f"strategy plugin contract is missing: {contract}")
    return contract.resolve()


def _display_contract_path(repo_root: Path, contract: Path) -> str:
    try:
        return contract.relative_to(repo_root).as_posix()
    except ValueError:
        return contract.as_posix()


def _artifact_versions(strategy_artifacts: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for sid, raw in sorted(dict(strategy_artifacts or {}).items(), key=lambda item: str(item[0])):
        row = _artifact_payload(raw)
        result[str(sid).upper().strip()] = {
            "artifact_hash": str(row.get("artifact_hash") or ""),
            "artifact_stage": str(row.get("artifact_stage") or row.get("stage") or ""),
            "source_fingerprint": str(row.get("source_fingerprint") or ""),
        }
    return result


def _artifact_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    metadata = dict(getattr(raw, "metadata", {}) or {})
    return {
        "artifact_hash": getattr(raw, "artifact_hash", ""),
        "artifact_stage": metadata.get("artifact_stage", ""),
        "source_fingerprint": getattr(raw, "source_fingerprint", ""),
    }


def _iso_utc(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if value not in (None, ""):
        raw = str(value)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw
        return _iso_utc(parsed)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _host_fingerprint() -> str:
    raw = "|".join(
        str(item or "")
        for item in (
            platform.node(),
            platform.system(),
            platform.machine(),
            os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        )
    )
    return canonical_json_hash({"host": raw})
