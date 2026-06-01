"""Adapters that export PaperSessionRecorder streams to canonical events."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .event_writer import JSONLEventWriter
from .family_snapshot import build_family_daily_snapshot
from .lineage import LineageContext, context_from_runtime, deployment_id_for, stable_hash


SESSION_STREAM_EVENT_TYPES: dict[str, str] = {
    "decision_stream.jsonl": "decision_event",
    "strategy_actions.jsonl": "strategy_action",
    "portfolio_arbitration.jsonl": "portfolio_rule",
    "oms_intents.jsonl": "oms_intent",
    "order_events.jsonl": "order",
    "fill_events.jsonl": "fill",
    "trade_outcomes.jsonl": "trade",
    "state_snapshots.jsonl": "position_snapshot",
    "subscription_events.jsonl": "market_data_subscription",
    "artifact_generation.jsonl": "config_snapshot",
}


class RuntimeAssistantExporter:
    """Fail-open exporter for active OLR/KALCB runtime evidence streams."""

    def __init__(self, data_dir: str | Path, *, lineage: LineageContext | None = None) -> None:
        self.base_lineage = lineage or context_from_runtime({}, data_source_id="runtime_session")
        self.current_lineage = self.base_lineage
        self.writer = JSONLEventWriter(data_dir, lineage=self.current_lineage)
        self._join_index: dict[str, dict[str, Any]] = {}

    def export_stream_row(
        self,
        filename: str,
        payload: Mapping[str, Any],
        *,
        session_root: str | Path,
        trade_date: date,
    ) -> dict[str, Any] | None:
        try:
            event_type = _event_type_for(filename, payload)
            if event_type is None:
                return None
            row = dict(payload or {})
            row.setdefault("session_root", str(session_root))
            row.setdefault("trade_date", trade_date.isoformat())
            row.setdefault("source_stream", filename)
            self._index_stream_row(filename, row)
            if event_type == "trade":
                row = self._enrich_trade(row)
            lineage = context_from_runtime(row, data_source_id=_data_source_for(event_type)).with_overrides(
                deployment_id=self.current_lineage.deployment_id,
                config_version=self.current_lineage.config_version,
                portfolio_config_version=self.current_lineage.portfolio_config_version,
                risk_config_version=self.current_lineage.risk_config_version,
                allocation_version=self.current_lineage.allocation_version,
                strategy_registry_version=self.current_lineage.strategy_registry_version,
                code_sha=self.current_lineage.code_sha,
                kis_resource_plan_hash=row.get("kis_resource_plan_hash") or self.current_lineage.kis_resource_plan_hash,
                portfolio_policy_hash=row.get("portfolio_policy_hash") or self.current_lineage.portfolio_policy_hash,
            )
            payload_key = _payload_key(row)
            event = self.writer.write(
                event_type,
                _canonical_runtime_payload(event_type, row),
                payload_key=payload_key,
                exchange_timestamp=_timestamp_for(row),
                lineage=lineage,
                scope=_scope_for(event_type, row),
            )
            if event_type == "fill":
                self._write_fill_context_snapshots(row, lineage=lineage)
            return event
        except Exception:
            return None

    def export_manifest(self, manifest: Mapping[str, Any], *, session_root: str | Path, trade_date: date) -> None:
        try:
            payload = dict(manifest or {})
            payload.setdefault("session_root", str(session_root))
            payload.setdefault("trade_date", trade_date.isoformat())
            if "closeout_reason" in payload:
                payload.setdefault("end_of_day_positions", _read_json_file(Path(session_root) / "end_of_day_positions.json"))
                payload.setdefault("session_rollup", _session_rollup(Path(session_root)))
            strategy_ids = tuple(str(item).upper().strip() for item in payload.get("strategy_ids") or ())
            versions = {
                "config_version": stable_hash(payload.get("strategy_configs") or {}),
                "portfolio_config_version": stable_hash(payload.get("portfolio_policy_config") or {}),
                "allocation_version": stable_hash(_allocations_from_positions(payload.get("initial_positions"))),
                "strategy_registry_version": stable_hash(
                    {
                        "strategy_ids": strategy_ids,
                        "mode": payload.get("mode"),
                        "staged_artifacts": payload.get("staged_artifacts"),
                    }
                ),
                "kis_resource_plan_hash": str(payload.get("kis_resource_plan_hash") or ""),
            }
            deployment_id = self.base_lineage.deployment_id or deployment_id_for({"manifest": versions, "code_sha": self.base_lineage.code_sha})
            lineage = self.base_lineage.with_overrides(
                deployment_id=deployment_id,
                config_version=versions["config_version"],
                portfolio_config_version=versions["portfolio_config_version"],
                allocation_version=versions["allocation_version"],
                strategy_registry_version=versions["strategy_registry_version"],
                kis_resource_plan_hash=versions["kis_resource_plan_hash"],
                portfolio_policy_hash=str(payload.get("portfolio_policy_hash") or self.base_lineage.portfolio_policy_hash),
            )
            self.current_lineage = lineage
            self.writer.lineage = lineage
            self.writer.write(
                "deployment",
                {
                    "record_type": "deployment",
                    "deployment_id": deployment_id,
                    "mode": payload.get("mode", ""),
                    "strategy_ids": list(strategy_ids),
                    "status": "started" if "closeout_reason" not in payload else "closed",
                    "source": "PaperSessionRecorder.write_manifest",
                    "artifact_hashes": _artifact_hashes(payload),
                    "source_fingerprints": _source_fingerprints(payload),
                    **payload,
                },
                payload_key=f"{deployment_id}:{payload.get('generated_at', '')}:{payload.get('closeout_reason', 'manifest')}",
                exchange_timestamp=payload.get("generated_at"),
                lineage=lineage,
                scope="portfolio",
            )
            self.writer.write(
                "config_snapshot",
                {
                    "record_type": "config_snapshot",
                    "deployment_id": deployment_id,
                    **versions,
                    "strategy_configs": payload.get("strategy_configs") or {},
                    "portfolio_policy_config": payload.get("portfolio_policy_config") or {},
                    "sector_map_hash": stable_hash(payload.get("sector_map") or {}),
                    "staged_artifacts": payload.get("staged_artifacts") or [],
                    "kis_resource_plan_path": payload.get("kis_resource_plan_path", ""),
                },
                payload_key=f"{deployment_id}:{versions['config_version']}",
                exchange_timestamp=payload.get("generated_at"),
                lineage=lineage,
                scope="portfolio",
            )
            if "initial_account_state" in payload or "initial_positions" in payload:
                self._write_runtime_snapshots(payload, lineage=lineage, reason="runtime_session_start")
            if "closeout_reason" in payload:
                self._write_closeout_events(payload, lineage=lineage, deployment_id=deployment_id, session_root=Path(session_root))
        except Exception:
            return

    def export_resource_plan(self, payload: Mapping[str, Any], *, session_root: str | Path, trade_date: date) -> None:
        try:
            row = dict(payload or {})
            row.setdefault("record_type", "resource_plan")
            row.setdefault("session_root", str(session_root))
            row.setdefault("trade_date", trade_date.isoformat())
            lineage = self.current_lineage.with_overrides(kis_resource_plan_hash=row.get("plan_hash") or "")
            self.current_lineage = lineage
            self.writer.lineage = lineage
            self.writer.write(
                "resource_plan",
                row,
                payload_key=str(row.get("plan_hash") or stable_hash(row)),
                exchange_timestamp=row.get("generated_at"),
                lineage=lineage,
                scope="portfolio",
            )
        except Exception:
            return

    def _write_runtime_snapshots(self, payload: Mapping[str, Any], *, lineage: LineageContext, reason: str) -> None:
        raw_positions = payload.get("end_of_day_positions") if reason == "runtime_session_closeout" else payload.get("initial_positions")
        positions = _positions_from_manifest(raw_positions)
        allocations = _allocations_from_positions(raw_positions)
        account = dict(payload.get("initial_account_state") or {}) if isinstance(payload.get("initial_account_state"), Mapping) else {}
        timestamp = payload.get("generated_at") or payload.get("closeout_generated_at")
        self.writer.write(
            "portfolio_snapshot",
            {
                "record_type": "portfolio_snapshot",
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "portfolio_id": lineage.portfolio_id,
                "account_alias": lineage.account_alias,
                "equity_krw": _float_or_zero(account.get("equity", account.get("total_equity"))),
                "buyable_cash_krw": _float_or_zero(account.get("buyable_cash", account.get("cash"))),
                "positions_count": len(positions),
                "working_orders_count": sum(len(row.get("working_orders") or ()) for row in positions),
                "positions": positions,
            },
            payload_key=f"{reason}:{stable_hash({'account': account, 'positions': positions})}",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="portfolio",
        )
        self.writer.write(
            "position_snapshot",
            {
                "record_type": "position_snapshot",
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "positions": positions,
            },
            payload_key=f"{reason}:{stable_hash(positions)}",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="oms",
        )
        self.writer.write(
            "allocation_snapshot",
            {
                "record_type": "allocation_snapshot",
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "allocations": allocations,
            },
            payload_key=f"{reason}:{stable_hash(allocations)}",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="oms",
        )

    def _write_closeout_events(self, payload: Mapping[str, Any], *, lineage: LineageContext, deployment_id: str, session_root: Path) -> None:
        timestamp = payload.get("closeout_generated_at") or payload.get("generated_at")
        key_base = f"{deployment_id}:{payload.get('trade_date', '')}:{payload.get('closeout_reason', '')}"
        session_rollup = dict(payload.get("session_rollup") or _session_rollup(session_root))
        closeout = {
            "record_type": "session_closeout",
            "deployment_id": deployment_id,
            "status": payload.get("hash_contract_status", ""),
            "source": "PaperSessionRecorder.close_session",
            "session_rollup": session_rollup,
            **dict(payload),
        }
        self.writer.write("session_closeout", closeout, payload_key=key_base, exchange_timestamp=timestamp, lineage=lineage, scope="portfolio")
        daily = {
            "record_type": "daily_snapshot",
            "deployment_id": deployment_id,
            "trade_date": payload.get("trade_date", ""),
            "generated_at": timestamp or datetime.now(timezone.utc).isoformat(),
            "hash_contract_version": payload.get("hash_contract_version", ""),
            "hash_contract_status": payload.get("hash_contract_status", ""),
            "expected_hashes_complete": bool(payload.get("expected_hashes_complete")),
            "session_metrics": dict(payload.get("session_metrics") or {}),
            "session_rollup": session_rollup,
            "expected_hashes": dict(payload.get("expected_hashes") or {}),
            "closeout_missing_required_files": list(payload.get("closeout_missing_required_files") or ()),
            "closeout_missing_required_dirs": list(payload.get("closeout_missing_required_dirs") or ()),
            "closeout_missing_artifact_evidence": list(payload.get("closeout_missing_artifact_evidence") or ()),
            "closeout_missing_resource_plan": list(payload.get("closeout_missing_resource_plan") or ()),
            "closeout_missing_hash_groups": list(payload.get("closeout_missing_hash_groups") or ()),
        }
        self.writer.write("daily_snapshot", daily, payload_key=f"{key_base}:daily", exchange_timestamp=timestamp, lineage=lineage, scope="portfolio")
        trade_date = _trade_date(payload)
        family = build_family_daily_snapshot(
            trade_date=trade_date,
            family_id=lineage.family_id,
            strategy_summaries=_strategy_summaries(payload),
            portfolio_summary={**daily, "strategy_ids": list(payload.get("strategy_ids") or ()), **session_rollup.get("portfolio", {})},
            replay_parity_status=str(payload.get("promotion_status") or payload.get("hash_contract_status") or ""),
        )
        self.writer.write("family_daily_snapshot", family, payload_key=f"{key_base}:family", exchange_timestamp=timestamp, lineage=lineage, scope="family")
        self._write_runtime_snapshots(payload, lineage=lineage, reason="runtime_session_closeout")
        self.writer.write(
            "resource_plan",
            {
                "record_type": "resource_plan",
                "trade_date": payload.get("trade_date", ""),
                "plan_hash": payload.get("kis_resource_plan_hash", ""),
                "path": payload.get("kis_resource_plan_path", ""),
                "closeout_status": payload.get("hash_contract_status", ""),
                "source": "PaperSessionRecorder.close_session",
            },
            payload_key=str(payload.get("kis_resource_plan_hash") or f"{key_base}:resource_plan"),
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="portfolio",
        )

    def _index_stream_row(self, filename: str, row: Mapping[str, Any]) -> None:
        compact = _join_payload(row)
        if filename == "trade_outcomes.jsonl":
            return
        for value in _join_refs(compact):
            existing = dict(self._join_index.get(value) or {})
            existing.update({key: item for key, item in compact.items() if item not in (None, "", [], {})})
            self._join_index[value] = existing

    def _enrich_trade(self, row: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        for ref in _join_refs(row):
            for key, value in dict(self._join_index.get(ref) or {}).items():
                enriched.setdefault(key, value)
        trade_id = str(enriched.get("trade_id") or "")
        if not trade_id:
            trade_id = stable_hash(
                {
                    "strategy_id": enriched.get("strategy_id"),
                    "symbol": enriched.get("symbol"),
                    "entry_order_id": enriched.get("entry_order_id"),
                    "exit_order_id": enriched.get("exit_order_id") or enriched.get("order_id"),
                    "exit_time": enriched.get("exit_time"),
                }
            )
            enriched["trade_id"] = trade_id
        enriched.setdefault("deployment_id", self.current_lineage.deployment_id)
        enriched.setdefault("config_version", self.current_lineage.config_version)
        enriched.setdefault("portfolio_config_version", self.current_lineage.portfolio_config_version)
        enriched.setdefault("risk_config_version", self.current_lineage.risk_config_version)
        enriched.setdefault("allocation_version", self.current_lineage.allocation_version)
        enriched.setdefault("kis_resource_plan_hash", self.current_lineage.kis_resource_plan_hash)
        enriched.setdefault("portfolio_policy_hash", self.current_lineage.portfolio_policy_hash)
        required = (
            "decision_ref",
            "action_ref",
            "portfolio_decision_ref",
            "intent_id",
            "order_id",
            "trade_id",
            "artifact_hash",
            "config_version",
            "deployment_id",
            "kis_resource_plan_hash",
        )
        enriched["join_completeness"] = {key: bool(enriched.get(key)) for key in required}
        return enriched

    def _write_fill_context_snapshots(self, row: Mapping[str, Any], *, lineage: LineageContext) -> None:
        context = row.get("portfolio_context_after")
        if not isinstance(context, Mapping):
            return
        timestamp = _timestamp_for(row)
        key_base = str(row.get("event_ref") or row.get("order_id") or stable_hash(row))
        common = {
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "reason": "fill_applied",
            "source_stream": "fill_events.jsonl",
            "strategy_id": row.get("strategy_id", ""),
            "symbol": row.get("symbol") or dict(row.get("event") or {}).get("symbol", ""),
            "order_id": row.get("order_id") or dict(row.get("event") or {}).get("order_id", ""),
            "intent_id": row.get("intent_id", ""),
            "portfolio_decision_ref": row.get("portfolio_decision_ref", ""),
        }
        self.writer.write(
            "portfolio_snapshot",
            {**dict(context), "record_type": "portfolio_snapshot", **common},
            payload_key=f"{key_base}:portfolio_after_fill",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="portfolio",
        )
        self.writer.write(
            "position_snapshot",
            {"record_type": "position_snapshot", **common, "positions": list(context.get("positions") or ())},
            payload_key=f"{key_base}:positions_after_fill",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="oms",
        )
        self.writer.write(
            "allocation_snapshot",
            {"record_type": "allocation_snapshot", **common, "allocations": list(context.get("allocations") or ())},
            payload_key=f"{key_base}:allocations_after_fill",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="oms",
        )


def _event_type_for(filename: str, payload: Mapping[str, Any]) -> str | None:
    record_type = str(payload.get("record_type") or "")
    if filename == "decision_stream.jsonl" and record_type in {"runtime_event_input", "runtime_no_action", "decision_event"}:
        return "decision_event"
    if filename == "portfolio_arbitration.jsonl":
        return "portfolio_rule"
    return SESSION_STREAM_EVENT_TYPES.get(filename)


def _canonical_runtime_payload(event_type: str, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["assistant_event_type"] = event_type
    if event_type == "portfolio_rule":
        payload.setdefault("decision_category", _portfolio_category(str(payload.get("reason_code") or payload.get("decision") or "")))
    if event_type == "decision_event" and payload.get("record_type") == "runtime_no_action":
        payload.setdefault("decision_code", "no_action")
    if event_type == "trade":
        payload.setdefault("event_type", "trade")
        payload.setdefault("schema_version", "trade_event_v2")
        payload.setdefault("currency", "KRW")
        payload.setdefault("exchange", "KRX")
    return payload


def _portfolio_category(reason: str) -> str:
    mapping = {
        "accepted": "accepted",
        "accepted_exit_reduces_exposure": "accepted",
        "resized_to_capacity": "portfolio_resized",
        "resized_to_existing_exposure": "portfolio_resized",
        "zero_quantity_or_notional": "sizing_block",
        "missing_or_zero_account_state": "account_state_gap",
        "duplicate_symbol_conflict": "symbol_collision",
        "capital_or_exposure_limit": "risk_cap_hit",
        "capacity_below_min_quantity": "risk_cap_hit",
        "exit_capacity_below_min_quantity": "position_state_gap",
        "unsupported_short_or_unmatched_exit": "position_state_gap",
    }
    return mapping.get(reason, reason or "unknown")


def _payload_key(row: Mapping[str, Any]) -> str:
    for key in (
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "intent_id",
        "idempotency_key",
        "order_id",
        "broker_order_id",
        "execution_id",
        "trade_id",
        "state_hash",
        "event_ref",
        "kis_resource_plan_hash",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return stable_hash(row)


def _join_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    nested_event = payload.get("event")
    if isinstance(nested_event, Mapping):
        payload.update({key: value for key, value in nested_event.items() if key not in payload and value not in (None, "")})
        metadata = nested_event.get("metadata")
        if isinstance(metadata, Mapping):
            payload.update({key: value for key, value in metadata.items() if key not in payload and value not in (None, "")})
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload.update({key: value for key, value in metadata.items() if key not in payload and value not in (None, "")})
    if payload.get("source_artifact_hash") and not payload.get("artifact_hash"):
        payload["artifact_hash"] = payload["source_artifact_hash"]
    if payload.get("broker_order_id") and not payload.get("kis_order_id"):
        payload["kis_order_id"] = payload["broker_order_id"]
    return payload


def _join_refs(row: Mapping[str, Any]) -> tuple[str, ...]:
    payload = _join_payload(row)
    refs: list[str] = []
    for key in (
        "trade_id",
        "event_ref",
        "decision_ref",
        "action_ref",
        "portfolio_decision_ref",
        "provisional_order_ref",
        "intent_id",
        "idempotency_key",
        "order_id",
        "broker_order_id",
        "original_order_id",
        "kis_order_id",
        "entry_order_id",
        "exit_order_id",
        "exit_fill_id",
        "kis_exec_id",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            refs.append(str(value))
    return tuple(dict.fromkeys(refs))


def _timestamp_for(row: Mapping[str, Any]) -> str | datetime | None:
    event = row.get("event")
    event_mapping = event if isinstance(event, Mapping) else {}
    return (
        row.get("timestamp")
        or row.get("event_time")
        or row.get("recorded_at")
        or row.get("order_submitted_at")
        or row.get("oms_received_at")
        or event_mapping.get("timestamp")
        or datetime.now(timezone.utc)
    )


def _scope_for(event_type: str, row: Mapping[str, Any]) -> str:
    if event_type in {"risk_decision", "oms_intent", "order", "fill", "position_snapshot", "allocation_snapshot"}:
        return "oms"
    if event_type in {"portfolio_rule", "portfolio_snapshot", "resource_plan", "market_data_subscription"}:
        return "portfolio"
    return "strategy"


def _data_source_for(event_type: str) -> str:
    if event_type in {"oms_intent", "order", "fill"}:
        return "postgres_oms"
    if event_type == "market_data_subscription":
        return "kis_websocket"
    return "runtime_session"


def _artifact_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in payload.get("staged_artifacts") or ():
        item = dict(row or {})
        sid = str(item.get("strategy_id") or "").upper().strip()
        if sid and item.get("artifact_hash"):
            result[sid] = str(item.get("artifact_hash"))
    return result


def _source_fingerprints(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in payload.get("staged_artifacts") or ():
        item = dict(row or {})
        sid = str(item.get("strategy_id") or "").upper().strip()
        if sid and item.get("source_fingerprint"):
            result[sid] = str(item.get("source_fingerprint"))
    return result


def _positions_from_manifest(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "positions" in raw:
            return _positions_from_manifest(raw.get("positions"))
        rows = []
        for symbol, value in sorted(raw.items(), key=lambda item: str(item[0])):
            row = dict(value or {}) if isinstance(value, Mapping) else {"value": value}
            row.setdefault("symbol", str(row.get("symbol") or symbol).zfill(6))
            rows.append(row)
        return rows
    if isinstance(raw, (list, tuple)):
        rows = []
        for value in raw:
            row = dict(value or {}) if isinstance(value, Mapping) else {"value": value}
            if row.get("symbol") not in (None, ""):
                row["symbol"] = str(row["symbol"]).zfill(6)
            rows.append(row)
        return rows
    return []


def _allocations_from_positions(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in _positions_from_manifest(raw):
        symbol = str(pos.get("symbol") or "").zfill(6)
        allocations = pos.get("allocations") or pos.get("strategy_allocations") or {}
        if isinstance(allocations, Mapping):
            iterator = sorted(allocations.items(), key=lambda item: str(item[0]))
            for strategy_id, value in iterator:
                row = dict(value or {}) if isinstance(value, Mapping) else {"qty": value}
                rows.append({"symbol": symbol, "strategy_id": str(strategy_id).upper().strip(), **row})
        elif isinstance(allocations, (list, tuple)):
            for value in allocations:
                row = dict(value or {}) if isinstance(value, Mapping) else {"qty": value}
                row.setdefault("symbol", symbol)
                row["strategy_id"] = str(row.get("strategy_id") or "").upper().strip()
                rows.append(row)
    return rows


def _trade_date(payload: Mapping[str, Any]) -> date:
    raw = payload.get("trade_date")
    if isinstance(raw, date):
        return raw
    if raw:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()


def _strategy_summaries(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = dict(payload.get("session_metrics") or {})
    rollup = dict(payload.get("session_rollup") or {})
    by_strategy = dict(rollup.get("strategies") or {})
    explicit = metrics.get("strategy_summaries")
    if isinstance(explicit, Mapping):
        summaries = {str(key).upper().strip(): dict(value or {}) for key, value in explicit.items()}
        for sid, row in by_strategy.items():
            summaries.setdefault(str(sid).upper().strip(), {}).update(dict(row or {}))
        return summaries
    return {
        str(strategy_id).upper().strip(): _strategy_summary_for(str(strategy_id).upper().strip(), payload, metrics, by_strategy)
        for strategy_id in payload.get("strategy_ids") or ()
    }


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def _session_rollup(session_root: Path) -> dict[str, Any]:
    trades = _read_jsonl(session_root / "trade_outcomes.jsonl")
    fills = _read_jsonl(session_root / "fill_events.jsonl")
    portfolio = _read_jsonl(session_root / "portfolio_arbitration.jsonl")
    orders = _read_jsonl(session_root / "order_events.jsonl")
    subscriptions = _read_jsonl(session_root / "subscription_events.jsonl")
    by_strategy: dict[str, dict[str, Any]] = {}
    for row in trades:
        sid = str(row.get("strategy_id") or "").upper().strip()
        if not sid:
            continue
        summary = by_strategy.setdefault(sid, _empty_strategy_summary())
        pnl = _float_or_zero(row.get("realized_pnl"))
        summary["total_trades"] += 1
        summary["realized_pnl"] += pnl
        summary["wins"] += 1 if pnl > 0 else 0
        summary["losses"] += 1 if pnl < 0 else 0
    for row in fills:
        sid = str(row.get("strategy_id") or _join_payload(row).get("strategy_id") or "").upper().strip()
        if not sid:
            continue
        summary = by_strategy.setdefault(sid, _empty_strategy_summary())
        summary["fills"] += 1
        payload = _join_payload(row)
        if bool(payload.get("inferred")):
            summary["inferred_fills"] += 1
        if str(payload.get("status") or "").upper() == "PARTIAL" or _float_or_zero(payload.get("qty")) < _float_or_zero(payload.get("order_qty")):
            summary["partial_fills"] += 1
    for row in portfolio:
        sid = str(row.get("strategy_id") or "").upper().strip()
        if not sid:
            continue
        summary = by_strategy.setdefault(sid, _empty_strategy_summary())
        decision = str(row.get("decision") or "").lower()
        if decision == "blocked":
            summary["blocked_portfolio_decisions"] += 1
        elif decision == "resized":
            summary["resized_portfolio_decisions"] += 1
    for row in orders:
        sid = str(row.get("strategy_id") or _join_payload(row).get("strategy_id") or "").upper().strip()
        if not sid:
            continue
        summary = by_strategy.setdefault(sid, _empty_strategy_summary())
        status = str(row.get("status") or "").upper()
        if status == "REJECTED":
            summary["oms_rejects"] += 1
        elif status == "DEFERRED":
            summary["oms_defers"] += 1
    resource_suppressions = sum(1 for row in subscriptions if str(row.get("action") or "").lower() == "suppressed")
    total_realized = sum(_float_or_zero(row.get("realized_pnl")) for row in by_strategy.values())
    return {
        "strategies": by_strategy,
        "portfolio": {
            "total_trades": len(trades),
            "fills": len(fills),
            "portfolio_decisions": len(portfolio),
            "orders": len(orders),
            "resource_plan_suppressions": resource_suppressions,
            "realized_pnl": total_realized,
        },
    }


def _empty_strategy_summary() -> dict[str, Any]:
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "realized_pnl": 0.0,
        "fills": 0,
        "partial_fills": 0,
        "inferred_fills": 0,
        "blocked_portfolio_decisions": 0,
        "resized_portfolio_decisions": 0,
        "oms_rejects": 0,
        "oms_defers": 0,
    }


def _strategy_summary_for(
    strategy_id: str,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    by_strategy: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(by_strategy.get(strategy_id) or {})
    prefix = strategy_id.lower()
    if f"{prefix}_trades" in metrics:
        row["total_trades"] = metrics.get(f"{prefix}_trades", row.get("total_trades", 0))
    elif "total_trades" in metrics and "total_trades" not in row:
        row["total_trades"] = metrics.get("total_trades", 0)
    if f"{prefix}_wins" in metrics:
        row["wins"] = metrics.get(f"{prefix}_wins", row.get("wins", 0))
    if f"{prefix}_losses" in metrics:
        row["losses"] = metrics.get(f"{prefix}_losses", row.get("losses", 0))
    row.setdefault("total_trades", 0)
    row.setdefault("wins", 0)
    row.setdefault("losses", 0)
    row["artifact_hash"] = _artifact_hashes(payload).get(strategy_id, "")
    row["source_fingerprint"] = _source_fingerprints(payload).get(strategy_id, "")
    return row
