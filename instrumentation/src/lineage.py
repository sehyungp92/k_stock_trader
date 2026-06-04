"""Canonical lineage context for assistant-facing telemetry."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_BOT_ID = "k_stock_trader"
DEFAULT_FAMILY_ID = "krx_equity"
DEFAULT_PORTFOLIO_ID = "olr_kalcb"
DEFAULT_ACCOUNT_ALIAS = "kis_primary"
DEFAULT_EXCHANGE = "KRX"
DEFAULT_ASSET_CLASS = "kr_equity"
DEFAULT_CURRENCY = "KRW"
DEFAULT_TIMEZONE = "Asia/Seoul"

SECRET_MARKERS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "APPKEY",
    "APP_KEY",
    "APPSECRET",
    "APP_SECRET",
    "HMAC",
    "DSN",
    "DATABASE_URL",
    "ACCOUNT_NO",
    "CANO",
    "ACNT",
)

_CODE_SHA_CACHE: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class LineageContext:
    """Stable identity and config lineage copied onto canonical events."""

    bot_id: str = DEFAULT_BOT_ID
    family_id: str = DEFAULT_FAMILY_ID
    portfolio_id: str = DEFAULT_PORTFOLIO_ID
    account_alias: str = DEFAULT_ACCOUNT_ALIAS
    strategy_id: str = ""
    data_source_id: str = "runtime_session"
    exchange: str = DEFAULT_EXCHANGE
    asset_class: str = DEFAULT_ASSET_CLASS
    currency: str = DEFAULT_CURRENCY
    timezone: str = DEFAULT_TIMEZONE
    strategy_version: str = ""
    config_version: str = ""
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    strategy_registry_version: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    experiment_id: str = ""
    variant_id: str = ""
    code_sha: str = ""
    proposal_ids: tuple[str, ...] = ()
    suggestion_ids: tuple[str, ...] = ()
    artifact_hash: str = ""
    source_fingerprint: str = ""
    candidate_hash: str = ""
    kis_resource_plan_hash: str = ""
    portfolio_policy_hash: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_empty: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["proposal_ids"] = list(self.proposal_ids)
        payload["suggestion_ids"] = list(self.suggestion_ids)
        extra = dict(payload.pop("extra", {}) or {})
        payload.update(extra)
        if include_empty:
            return payload
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", None, (), [], {})
        }

    def with_overrides(self, **overrides: Any) -> "LineageContext":
        allowed = set(LineageContext.__dataclass_fields__)
        clean = {key: value for key, value in overrides.items() if value is not None and key in allowed}
        return replace(self, **clean)

    def monthly_lineage_gaps(self, *, scope: str = "strategy") -> tuple[str, ...]:
        required = ["deployment_id", "code_sha"]
        if scope == "strategy":
            required.extend(["strategy_id", "family_id", "strategy_version", "config_version"])
        if scope == "family":
            required.extend(["family_id", "strategy_version", "config_version"])
        if scope in {"portfolio", "oms", "family"}:
            required.extend(["portfolio_id", "portfolio_config_version", "risk_config_version", "allocation_version"])
        if scope == "oms" and self.strategy_id and self.strategy_id != "_UNKNOWN_":
            required.extend(["strategy_version", "config_version"])
        return tuple(field_name for field_name in dict.fromkeys(required) if not getattr(self, field_name, ""))


def context_from_env(
    *,
    strategy_id: str = "",
    data_source_id: str = "runtime_session",
    family_id: str | None = None,
    portfolio_id: str | None = None,
    account_alias: str | None = None,
    deployment_id: str | None = None,
    code_sha: str | None = None,
    **overrides: Any,
) -> LineageContext:
    """Build lineage from non-secret environment variables plus overrides."""

    proposal_ids = _split_ids(os.environ.get("TRADING_ASSISTANT_PROPOSAL_IDS", ""))
    suggestion_ids = _split_ids(os.environ.get("TRADING_ASSISTANT_SUGGESTION_IDS", ""))
    return LineageContext(
        bot_id=os.environ.get("BOT_ID", DEFAULT_BOT_ID),
        family_id=family_id or os.environ.get("FAMILY_ID", DEFAULT_FAMILY_ID),
        portfolio_id=portfolio_id or os.environ.get("PORTFOLIO_ID", DEFAULT_PORTFOLIO_ID),
        account_alias=account_alias or _account_alias_from_env(),
        strategy_id=str(strategy_id or "").upper().strip(),
        data_source_id=data_source_id,
        strategy_version=os.environ.get("STRATEGY_VERSION", ""),
        config_version=os.environ.get("CONFIG_VERSION", ""),
        portfolio_config_version=os.environ.get("PORTFOLIO_CONFIG_VERSION", ""),
        risk_config_version=os.environ.get("RISK_CONFIG_VERSION", ""),
        allocation_version=os.environ.get("ALLOCATION_VERSION", ""),
        strategy_registry_version=os.environ.get("STRATEGY_REGISTRY_VERSION", ""),
        deployment_id=deployment_id or os.environ.get("DEPLOYMENT_ID", ""),
        parameter_set_id=os.environ.get("PARAMETER_SET_ID", ""),
        experiment_id=os.environ.get("EXPERIMENT_ID", ""),
        variant_id=os.environ.get("VARIANT_ID", ""),
        code_sha=code_sha or os.environ.get("CODE_SHA", "") or get_code_sha(),
        proposal_ids=proposal_ids,
        suggestion_ids=suggestion_ids,
    ).with_overrides(**overrides)


def context_from_runtime(
    payload: Mapping[str, Any] | None = None,
    *,
    strategy_id: str = "",
    data_source_id: str = "runtime_session",
    **overrides: Any,
) -> LineageContext:
    row = dict(payload or {})
    metadata = dict(row.get("metadata") or {})
    action = dict(row.get("action") or {})
    action_metadata = dict(action.get("metadata") or {})
    merged = {**metadata, **action_metadata}
    sid = strategy_id or row.get("strategy_id") or merged.get("strategy_id") or action.get("strategy_id") or ""
    return context_from_env(
        strategy_id=str(sid or "").upper().strip(),
        data_source_id=data_source_id,
        artifact_hash=(
            row.get("source_artifact_hash")
            or row.get("artifact_hash")
            or merged.get("source_artifact_hash")
            or merged.get("artifact_hash")
            or ""
        ),
        source_fingerprint=row.get("source_fingerprint") or merged.get("source_fingerprint") or "",
        candidate_hash=row.get("candidate_hash") or merged.get("candidate_hash") or "",
        kis_resource_plan_hash=row.get("kis_resource_plan_hash") or merged.get("kis_resource_plan_hash") or "",
        portfolio_policy_hash=row.get("portfolio_policy_hash") or merged.get("portfolio_policy_hash") or "",
        **overrides,
    )


def context_from_oms(
    payload: Mapping[str, Any] | None = None,
    *,
    strategy_id: str = "",
    data_source_id: str = "postgres_oms",
    **overrides: Any,
) -> LineageContext:
    row = dict(payload or {})
    metadata = dict(row.get("metadata") or {})
    sid = strategy_id or row.get("strategy_id") or metadata.get("strategy_id") or "_UNKNOWN_"
    return context_from_env(
        strategy_id=str(sid or "").upper().strip(),
        data_source_id=data_source_id,
        portfolio_id=os.environ.get("PORTFOLIO_ID", DEFAULT_PORTFOLIO_ID),
        account_alias=_account_alias_from_env(),
        **overrides,
    )


def stable_hash(payload: Any) -> str:
    import json

    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def deployment_id_for(parts: Mapping[str, Any] | Sequence[Any]) -> str:
    return f"deploy:{stable_hash(parts)}"


def redact_mapping(mapping: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a recursively redacted copy plus a sorted list of redacted keys."""

    redacted_keys: list[str] = []

    def redact(value: Any, path: str = "") -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if _looks_secret(key_text):
                    result[key_text] = "***REDACTED***"
                    redacted_keys.append(child_path)
                else:
                    result[key_text] = redact(item, child_path)
            return result
        if isinstance(value, (list, tuple)):
            return [redact(item, f"{path}[]") for item in value]
        return value

    return redact(mapping), sorted(dict.fromkeys(redacted_keys))


def get_code_sha(cwd: str | Path | None = None) -> str:
    resolved_cwd = str(Path(cwd or Path.cwd()).resolve())
    if resolved_cwd in _CODE_SHA_CACHE:
        return _CODE_SHA_CACHE[resolved_cwd]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=resolved_cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )
        value = result.stdout.strip()
    except Exception:
        value = ""
    _CODE_SHA_CACHE[resolved_cwd] = value
    return value


def _account_alias_from_env() -> str:
    for name in ("KIS_ACCOUNT_ALIAS", "ACCOUNT_ALIAS", "OMS_ACCOUNT_ALIAS"):
        value = os.environ.get(name)
        if value:
            return str(value)
    return DEFAULT_ACCOUNT_ALIAS


def _split_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").replace(";", ",").split(",") if item.strip())


def _looks_secret(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SECRET_MARKERS)
