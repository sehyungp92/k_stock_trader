"""Adapters that export PaperSessionRecorder streams to canonical events."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .event_writer import JSONLEventWriter
from .family_snapshot import build_family_daily_snapshot
from .lineage import LineageContext, context_from_runtime, stable_hash
from .runtime_lineage import (
    runtime_deployment_id,
    runtime_risk_config_from_manifest,
    runtime_versions_from_manifest,
    write_runtime_deployment_lineage,
)


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
        self._pending_resource_plans: list[tuple[dict[str, Any], str, date]] = []

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
                strategy_version=self.current_lineage.strategy_version,
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
            if event_type == "decision_event":
                self._write_market_snapshot_for_bar(row, lineage=lineage)
                self._write_indicator_snapshot(row, lineage=lineage)
                self._write_filter_decisions(row, lineage=lineage)
                self._write_strategy_gate_miss(row, lineage=lineage)
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
            risk_config = runtime_risk_config_from_manifest(payload)
            versions = runtime_versions_from_manifest(payload, strategy_ids=strategy_ids, risk_config=risk_config)
            if self.base_lineage.strategy_version:
                versions["strategy_version"] = self.base_lineage.strategy_version
            if self.base_lineage.risk_config_version:
                versions["risk_config_version"] = self.base_lineage.risk_config_version
            deployment_id = self.base_lineage.deployment_id or runtime_deployment_id(versions, code_sha=self.base_lineage.code_sha)
            lineage = self.base_lineage.with_overrides(
                deployment_id=deployment_id,
                strategy_version=versions["strategy_version"],
                config_version=versions["config_version"],
                portfolio_config_version=versions["portfolio_config_version"],
                risk_config_version=versions["risk_config_version"],
                allocation_version=versions["allocation_version"],
                strategy_registry_version=versions["strategy_registry_version"],
                kis_resource_plan_hash=versions["kis_resource_plan_hash"],
                portfolio_policy_hash=str(payload.get("portfolio_policy_hash") or self.base_lineage.portfolio_policy_hash),
            )
            self.current_lineage = lineage
            self.writer.lineage = lineage
            self._write_runtime_lineage_handoff(payload, strategy_ids=strategy_ids, versions=versions, risk_config=risk_config, deployment_id=deployment_id)
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
                    **versions,
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
                    "risk_config": risk_config,
                    "active_strategy_budget_status": _active_strategy_budget_status(strategy_ids, risk_config),
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
            self._flush_pending_resource_plans()
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
            if not self._resource_plan_ready(row):
                self._pending_resource_plans.append((row, str(session_root), trade_date))
                return
            self._write_resource_plan_row(row)
        except Exception:
            return

    def _flush_pending_resource_plans(self) -> None:
        pending = list(self._pending_resource_plans)
        self._pending_resource_plans.clear()
        for row, _session_root, _trade_date in pending:
            try:
                if not self._resource_plan_ready(row):
                    self._pending_resource_plans.append((row, _session_root, _trade_date))
                    continue
                self._write_resource_plan_row(row)
            except Exception:
                continue

    def _resource_plan_ready(self, row: Mapping[str, Any]) -> bool:
        if self.current_lineage.monthly_lineage_gaps(scope="portfolio"):
            return False
        plan_hash = str(row.get("plan_hash") or "")
        return not plan_hash or plan_hash == str(self.current_lineage.kis_resource_plan_hash or "")

    def _write_resource_plan_row(self, row: Mapping[str, Any]) -> None:
        plan_hash = str(row.get("plan_hash") or "")
        lineage = self.current_lineage.with_overrides(kis_resource_plan_hash=plan_hash or self.current_lineage.kis_resource_plan_hash)
        self.current_lineage = lineage
        self.writer.lineage = lineage
        self.writer.write(
            "resource_plan",
            dict(row),
            payload_key=str(row.get("plan_hash") or stable_hash(row)),
            exchange_timestamp=row.get("generated_at"),
            lineage=lineage,
            scope="portfolio",
        )

    def _write_market_snapshot_for_bar(self, row: Mapping[str, Any], *, lineage: LineageContext) -> None:
        if str(row.get("record_type") or "") != "runtime_event_input" or str(row.get("event_type") or "") != "bar":
            return
        payload = dict(row.get("payload") or {})
        if not payload:
            return
        bar_hash = str(row.get("bar_hash") or stable_hash(payload))
        timestamp = payload.get("timestamp") or row.get("timestamp")
        self.writer.write(
            "market_snapshot",
            {
                "record_type": "market_snapshot",
                "source": "PaperSessionRecorder.runtime_event_input",
                "strategy_id": row.get("strategy_id", ""),
                "symbol": str(payload.get("symbol") or row.get("symbol") or "").zfill(6),
                "timeframe": payload.get("timeframe", ""),
                "timestamp": timestamp,
                "bar_hash": bar_hash,
                "bar_id": bar_hash,
                "market_bar": payload,
                "session_root": row.get("session_root", ""),
                "trade_date": row.get("trade_date", ""),
                "event_ref": row.get("event_ref", ""),
            },
            payload_key=f"{row.get('strategy_id', '')}:{bar_hash}",
            exchange_timestamp=timestamp,
            lineage=lineage.with_overrides(strategy_id=str(row.get("strategy_id") or lineage.strategy_id or "").upper().strip()),
            scope="strategy",
        )

    def _write_filter_decisions(self, row: Mapping[str, Any], *, lineage: LineageContext) -> None:
        decisions = _decision_filters(row)
        if not decisions:
            return
        base_key = str(row.get("decision_ref") or stable_hash(row))
        timestamp = _timestamp_for(row)
        for index, decision in enumerate(decisions):
            payload = {
                "record_type": "filter_decision",
                "strategy_id": row.get("strategy_id", ""),
                "symbol": str(row.get("symbol") or row.get("pair") or "").zfill(6),
                "decision_ref": row.get("decision_ref", ""),
                "event_ref": row.get("event_ref", ""),
                "decision_code": row.get("decision_code", ""),
                "reason": row.get("reason", ""),
                "filter_index": index,
                **dict(decision),
            }
            self.writer.write(
                "filter_decision",
                payload,
                payload_key=f"{base_key}:filter:{index}",
                exchange_timestamp=timestamp,
                lineage=lineage,
                scope="strategy",
            )

    def _write_indicator_snapshot(self, row: Mapping[str, Any], *, lineage: LineageContext) -> None:
        snapshot = _decision_indicator_snapshot(row)
        if not snapshot:
            return
        timestamp = _timestamp_for(row)
        strategy_id = str(row.get("strategy_id") or lineage.strategy_id or "").upper().strip()
        symbol = str(row.get("symbol") or row.get("pair") or "").zfill(6)
        bar_id = str(row.get("bar_id") or row.get("event_ref") or "")
        payload = {
            "record_type": "indicator_snapshot",
            "strategy_id": strategy_id,
            "symbol": symbol,
            "pair": symbol,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp or ""),
            "decision_ref": row.get("decision_ref", ""),
            "event_ref": row.get("event_ref", ""),
            "bar_id": bar_id,
            **snapshot,
        }
        self.writer.write(
            "indicator_snapshot",
            payload,
            payload_key=f"{strategy_id}:{symbol}:{bar_id}:{row.get('decision_ref', '')}:indicators",
            exchange_timestamp=timestamp,
            lineage=lineage.with_overrides(strategy_id=strategy_id),
            scope="strategy",
        )

    def _write_strategy_gate_miss(self, row: Mapping[str, Any], *, lineage: LineageContext) -> None:
        if not _is_strategy_gate_reject(row):
            return
        metadata = _row_metadata(row)
        timestamp = _timestamp_for(row)
        logical_event_id = ":".join(
            str(part)
            for part in (
                str(row.get("strategy_id") or lineage.strategy_id or "").upper().strip(),
                str(row.get("symbol") or row.get("pair") or "").zfill(6),
                row.get("event_ref") or row.get("bar_id") or timestamp,
                row.get("decision_code") or "strategy_gate_reject",
                row.get("reason") or metadata.get("blocked_by") or "blocked",
            )
            if str(part or "").strip()
        )
        if not logical_event_id:
            logical_event_id = stable_hash(row)
        payload = {
            "record_type": "missed_opportunity",
            "event_type": "missed_opportunity",
            "schema_version": "missed_opportunity_v2",
            "logical_event_id": logical_event_id,
            "revision": 0,
            "strategy_id": str(row.get("strategy_id") or lineage.strategy_id or "").upper().strip(),
            "pair": str(row.get("symbol") or row.get("pair") or "").zfill(6),
            "side": _missed_side(row),
            "signal": str(row.get("reason") or metadata.get("signal") or ""),
            "signal_id": str(metadata.get("candidate_hash") or row.get("decision_ref") or logical_event_id),
            "signal_strength": _float_or_zero(metadata.get("candidate_score", metadata.get("signal_strength"))),
            "signal_time": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp or ""),
            "blocked_by": str(row.get("reason") or row.get("decision_code") or "strategy_gate_reject"),
            "block_reason": str(row.get("reason") or ""),
            "blocked_scope": "strategy_filter",
            "event_ref": row.get("event_ref", ""),
            "decision_ref": row.get("decision_ref", ""),
            "filter_decisions": _decision_filters(row),
            "gate_decisions": _gate_decisions(metadata),
            "decision": dict(row),
        }
        self.writer.write(
            "missed_opportunity",
            payload,
            payload_key=f"{logical_event_id}:rev:0",
            exchange_timestamp=timestamp,
            lineage=lineage.with_overrides(
                strategy_id=payload["strategy_id"],
                artifact_hash=metadata.get("source_artifact_hash") or lineage.artifact_hash,
                source_fingerprint=metadata.get("source_fingerprint") or lineage.source_fingerprint,
                candidate_hash=metadata.get("candidate_hash") or lineage.candidate_hash,
            ),
            logical_event_id=logical_event_id,
            revision=0,
            scope="strategy",
        )

    def _write_runtime_lineage_handoff(
        self,
        payload: Mapping[str, Any],
        *,
        strategy_ids: tuple[str, ...],
        versions: Mapping[str, str],
        risk_config: Mapping[str, Any],
        deployment_id: str,
    ) -> None:
        try:
            write_runtime_deployment_lineage(
                self.writer.data_dir,
                {
                    "record_type": "runtime_deployment_lineage",
                    "deployment_id": deployment_id,
                    "code_sha": self.base_lineage.code_sha,
                    "portfolio_id": self.current_lineage.portfolio_id,
                    "account_alias": self.current_lineage.account_alias,
                    "strategy_ids": list(strategy_ids),
                    **dict(versions),
                    "risk_config_version": versions.get("risk_config_version") or stable_hash(risk_config),
                    "portfolio_policy_hash": str(payload.get("portfolio_policy_hash") or self.current_lineage.portfolio_policy_hash),
                },
            )
        except Exception:
            return

    def _write_runtime_snapshots(self, payload: Mapping[str, Any], *, lineage: LineageContext, reason: str) -> None:
        raw_positions = payload.get("end_of_day_positions") if reason == "runtime_session_closeout" else payload.get("initial_positions")
        positions = _positions_from_manifest(raw_positions)
        allocations = _allocations_from_positions(raw_positions)
        account = _account_for_snapshot(payload, reason=reason)
        timestamp = _snapshot_timestamp(payload, reason=reason)
        self.writer.write(
            "portfolio_snapshot",
            {
                "record_type": "portfolio_snapshot",
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "portfolio_id": lineage.portfolio_id,
                "account_alias": lineage.account_alias,
                **_lineage_version_fields(lineage),
                **_runtime_portfolio_fields(positions, allocations, account),
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
                **_lineage_version_fields(lineage),
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
                **_lineage_version_fields(lineage),
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
        enriched.setdefault("strategy_version", self.current_lineage.strategy_version)
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
            {**dict(context), "record_type": "portfolio_snapshot", **_lineage_version_fields(lineage), **common},
            payload_key=f"{key_base}:portfolio_after_fill",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="portfolio",
        )
        self.writer.write(
            "position_snapshot",
            {"record_type": "position_snapshot", **_lineage_version_fields(lineage), **common, "positions": list(context.get("positions") or ())},
            payload_key=f"{key_base}:positions_after_fill",
            exchange_timestamp=timestamp,
            lineage=lineage,
            scope="oms",
        )
        self.writer.write(
            "allocation_snapshot",
            {"record_type": "allocation_snapshot", **_lineage_version_fields(lineage), **common, "allocations": list(context.get("allocations") or ())},
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


def _lineage_version_fields(lineage: LineageContext) -> dict[str, str]:
    keys = (
        "deployment_id",
        "strategy_version",
        "config_version",
        "portfolio_config_version",
        "risk_config_version",
        "allocation_version",
        "strategy_registry_version",
        "kis_resource_plan_hash",
        "portfolio_policy_hash",
    )
    return {
        key: str(getattr(lineage, key, "") or "")
        for key in keys
        if getattr(lineage, key, "") not in (None, "")
    }


def _active_strategy_budget_status(strategy_ids: tuple[str, ...], risk_config: Mapping[str, Any]) -> dict[str, str]:
    budgets = {str(key).upper().strip() for key in dict(risk_config.get("strategy_budgets") or {})}
    return {
        strategy_id: "configured" if strategy_id in budgets else "missing_uses_global_limits"
        for strategy_id in strategy_ids
        if strategy_id
    }


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


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    nested = row.get("payload")
    if isinstance(nested, Mapping):
        nested_metadata = nested.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata = {**dict(nested_metadata), **metadata}
    return metadata


def _decision_filters(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _row_metadata(row)
    result: list[dict[str, Any]] = []
    result.extend(_filter_rows(metadata.get("filter_decisions"), source="filter_decisions"))
    result.extend(_filter_rows(metadata.get("gate_decisions"), source="gate_decisions"))
    return result


_INDICATOR_METADATA_KEYS = (
    "bar_rvol",
    "rvol",
    "vwap_ret",
    "entry_vwap_ret",
    "avwap",
    "cpr",
    "daily_atr",
    "risk_per_share",
    "momentum_score",
    "flow_score",
    "accumulation_score",
    "candidate_score",
    "frontier_selection_score",
    "first30_vwap_ret",
    "first30_range_atr",
    "range_atr",
    "depth_atr",
    "afternoon_ret",
    "gap",
    "low_vs_prev_close",
    "selection_score",
    "expected_5m_volume",
    "average_30m_volume",
)


def _decision_indicator_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _row_metadata(row)
    explicit = metadata.get("indicator_snapshot")
    if isinstance(explicit, Mapping):
        snapshot = dict(explicit)
        indicators = _numeric_indicator_mapping(snapshot.get("indicators") or snapshot)
    else:
        indicators = _numeric_indicator_mapping(metadata.get("indicators"))
        if not indicators:
            indicators = {
                key: value
                for key in _INDICATOR_METADATA_KEYS
                for value in [_float_indicator_value(metadata.get(key))]
                if value is not None
            }
        snapshot = {}
    if not indicators:
        return {}
    context_keys = (
        "entry_type",
        "score_detail",
        "candidate_rank",
        "frontier_rank",
        "frontier_role",
        "frontier_selection_mode",
        "sector",
        "regime_tier",
        "entry_route",
        "trade_exit_plan",
        "afternoon_score_band_rule",
        "source_artifact_hash",
        "source_fingerprint",
        "candidate_hash",
    )
    context = {
        key: metadata.get(key)
        for key in context_keys
        if metadata.get(key) not in (None, "", {}, [])
    }
    context.update(dict(snapshot.get("context") or {}) if isinstance(snapshot.get("context"), Mapping) else {})
    return {
        "indicators": indicators,
        "signal_name": str(snapshot.get("signal_name") or metadata.get("signal") or row.get("reason") or row.get("decision_code") or ""),
        "signal_strength": _float_or_zero(snapshot.get("signal_strength", metadata.get("candidate_score", metadata.get("momentum_score")))),
        "decision": str(snapshot.get("decision") or row.get("decision_code") or ""),
        "context": context,
    }


def _numeric_indicator_mapping(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in dict(raw).items():
        numeric = _float_indicator_value(value)
        if numeric is not None:
            result[str(key)] = numeric
    return result


def _float_indicator_value(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gate_decisions(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _filter_rows(metadata.get("gate_decisions"), source="gate_decisions")


def _filter_rows(raw: Any, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for name, value in sorted(raw.items(), key=lambda item: str(item[0])):
            rows.append(_canonical_filter_payload(value, source=source, name=str(name)))
        return rows
    if isinstance(raw, (list, tuple)):
        for index, value in enumerate(raw):
            rows.append(_canonical_filter_payload(value, source=source, index=index))
    return rows


def _canonical_filter_payload(raw: Any, *, source: str = "", name: str = "", index: int = 0) -> dict[str, Any]:
    row = dict(raw) if isinstance(raw, Mapping) else {"passed": bool(raw), "actual_value": raw}
    filter_name = str(row.get("filter_name") or row.get("gate_name") or row.get("name") or name or f"filter_{index}")
    actual = row.get("actual_value", row.get("actual", row.get("observed")))
    threshold = row.get("threshold")
    margin = row.get("margin", row.get("margin_pct", row.get("distance_to_threshold")))
    passed = row.get("passed")
    if passed is None and row.get("result") not in (None, ""):
        passed = str(row.get("result")).lower() in {"pass", "passed", "true", "allow", "approved"}
    payload = {
        "filter_name": filter_name,
        "gate_name": str(row.get("gate_name") or filter_name),
        "filter_source": str(row.get("filter_source") or source or ""),
        "threshold": threshold,
        "threshold_operator": str(row.get("threshold_operator") or row.get("operator") or ""),
        "actual_value": actual,
        "passed": bool(passed) if passed is not None else False,
        "margin": margin,
        "margin_pct": row.get("margin_pct", margin if str(row.get("margin_type") or "").lower() == "pct" else None),
        "input_refs": _string_list(row.get("input_refs") or row.get("inputs") or row.get("input_ref")),
        "applicable": bool(row.get("applicable", True)),
        "raw_filter_decision": row,
    }
    for key, value in row.items():
        if key not in payload and key not in {"filter_decision", "raw_filter_decision"}:
            payload.setdefault(key, value)
    return payload


def _string_list(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if item not in (None, "")]
    return [str(raw)]


def _is_strategy_gate_reject(row: Mapping[str, Any]) -> bool:
    if str(row.get("record_type") or "") != "decision_event":
        return False
    decision_code = str(row.get("decision_code") or "").lower()
    if "reject" not in decision_code and "block" not in decision_code:
        return False
    filters = _decision_filters(row)
    if not filters:
        return False
    return any(item.get("passed") is False or str(item.get("result") or "").lower() in {"reject", "blocked", "fail"} for item in filters)


def _missed_side(row: Mapping[str, Any]) -> str:
    for action in row.get("actions") or ():
        if isinstance(action, Mapping):
            action_type = str(action.get("action_type") or action.get("type") or "")
            if "exit" in action_type.lower() or str(action.get("side") or "").upper() == "SELL":
                return "EXIT"
    return "LONG"


def _positions_from_manifest(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "positions" in raw:
            return _positions_from_manifest(raw.get("positions"))
        rows = []
        for symbol, value in sorted(raw.items(), key=lambda item: str(item[0])):
            row = dict(value or {}) if isinstance(value, Mapping) else {"value": value}
            rows.append(_normalize_position_row(row, symbol=symbol))
        return rows
    if isinstance(raw, (list, tuple)):
        rows = []
        for value in raw:
            row = dict(value or {}) if isinstance(value, Mapping) else {"value": value}
            rows.append(_normalize_position_row(row))
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
                row.setdefault("cost_basis", pos.get("avg_price", 0.0))
                rows.append({"symbol": symbol, "strategy_id": str(strategy_id).upper().strip(), **row})
        elif isinstance(allocations, (list, tuple)):
            for value in allocations:
                row = dict(value or {}) if isinstance(value, Mapping) else {"qty": value}
                row.setdefault("symbol", symbol)
                row["strategy_id"] = str(row.get("strategy_id") or "").upper().strip()
                row.setdefault("cost_basis", pos.get("avg_price", 0.0))
                rows.append(row)
    return rows


def _normalize_position_row(row: Mapping[str, Any], *, symbol: Any = "") -> dict[str, Any]:
    data = dict(row or {})
    if data.get("symbol") not in (None, ""):
        data["symbol"] = str(data["symbol"]).zfill(6)
    elif symbol not in (None, ""):
        data["symbol"] = str(symbol).zfill(6)
    qty = _position_qty(data)
    avg_price = _position_price(data)
    data.setdefault("real_qty", qty)
    data.setdefault("avg_price", avg_price)
    allocations = data.get("allocations") or data.get("strategy_allocations")
    strategy_id = str(data.get("strategy_id") or "").upper().strip()
    if not allocations and strategy_id and qty > 0:
        data["allocations"] = {
            strategy_id: {
                "strategy_id": strategy_id,
                "qty": qty,
                "cost_basis": avg_price,
                "entry_ts": data.get("entry_time", ""),
            }
        }
    return data


def _account_for_snapshot(payload: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    keys = (
        ("end_of_day_account_state", "final_account_state", "account_state", "initial_account_state")
        if reason == "runtime_session_closeout"
        else ("initial_account_state", "account_state")
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _snapshot_timestamp(payload: Mapping[str, Any], *, reason: str) -> Any:
    if reason == "runtime_session_closeout":
        return payload.get("closeout_generated_at") or payload.get("generated_at")
    return payload.get("generated_at") or payload.get("closeout_generated_at")


def _runtime_portfolio_fields(
    positions: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    account: Mapping[str, Any],
) -> dict[str, Any]:
    equity = _float_or_zero(account.get("equity", account.get("total_equity")))
    buyable_cash = _float_or_zero(account.get("buyable_cash", account.get("cash")))
    symbol_exposures: dict[str, dict[str, Any]] = {}
    sector_exposures: dict[str, dict[str, Any]] = {}
    strategy_exposures: dict[str, dict[str, Any]] = {}
    gross = 0.0
    allocation_qty_by_symbol: dict[str, int] = {}
    for allocation in allocations:
        symbol = str(allocation.get("symbol") or "").zfill(6)
        strategy_id = str(allocation.get("strategy_id") or "").upper().strip()
        qty = max(_int_or_zero(allocation.get("qty")), 0)
        price = max(_float_or_zero(allocation.get("cost_basis")), 0.0)
        allocation_qty_by_symbol[symbol] = allocation_qty_by_symbol.get(symbol, 0) + qty
        if strategy_id:
            row = strategy_exposures.setdefault(strategy_id, {"qty": 0, "notional_krw": 0.0, "symbols_count": 0})
            row["qty"] += qty
            row["notional_krw"] += qty * price
            row["symbols_count"] += 1 if qty > 0 else 0
    for position in positions:
        symbol = str(position.get("symbol") or "").zfill(6)
        qty = max(_position_qty(position), 0)
        price = max(_position_price(position), 0.0)
        notional = qty * price
        gross += abs(notional)
        sector = str(position.get("sector") or "UNKNOWN").upper().strip() or "UNKNOWN"
        allocated_qty = allocation_qty_by_symbol.get(symbol, 0)
        drift = qty - allocated_qty
        symbol_exposures[symbol] = {
            "qty": qty,
            "allocated_qty": allocated_qty,
            "avg_price": price,
            "notional_krw": notional,
            "sector": sector,
            "allocation_drift": drift,
            "frozen": bool(position.get("frozen", False)),
        }
        sector_row = sector_exposures.setdefault(sector, {"qty": 0, "notional_krw": 0.0, "symbols_count": 0})
        sector_row["qty"] += qty
        sector_row["notional_krw"] += notional
        sector_row["symbols_count"] += 1 if qty > 0 else 0
    return {
        "equity_krw": equity,
        "buyable_cash_krw": buyable_cash,
        "daily_pnl_krw": _float_or_zero(account.get("daily_pnl")),
        "daily_pnl_pct": _float_or_zero(account.get("daily_pnl_pct")),
        "gross_exposure_krw": gross,
        "gross_exposure_pct": gross / equity if equity > 0 else 0.0,
        "positions_count": len(positions),
        "working_orders_count": sum(_working_order_count(row) for row in positions),
        "allocation_drift_count": sum(1 for row in symbol_exposures.values() if row["allocation_drift"] != 0 or row["frozen"]),
        "sector_exposures": sector_exposures,
        "strategy_exposures": strategy_exposures,
        "symbol_exposures": symbol_exposures,
        "pending_reservations": {},
    }


def _position_qty(row: Mapping[str, Any]) -> int:
    return _int_or_zero(row.get("real_qty", row.get("qty_open", row.get("qty", 0))))


def _position_price(row: Mapping[str, Any]) -> float:
    return _float_or_zero(row.get("avg_price", row.get("entry_price", row.get("price", 0.0))))


def _working_order_count(row: Mapping[str, Any]) -> int:
    working_orders = row.get("working_orders")
    if isinstance(working_orders, (list, tuple)):
        return len(working_orders)
    return _int_or_zero(row.get("working_order_count", row.get("working_orders_count", 0)))


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
