"""Assistant-facing instrumentation event contract constants."""

from __future__ import annotations

from typing import Mapping


CANONICAL_SCHEMA_VERSION = "assistant_event_v1"


EVENT_DIRS: dict[str, str] = {
    "trade": "trades",
    "missed_opportunity": "missed",
    "order": "orders",
    "fill": "fills",
    "filter_decision": "filter_decisions",
    "indicator_snapshot": "indicators",
    "market_snapshot": "snapshots",
    "orderbook_context": "orderbook",
    "heartbeat": "heartbeats",
    "daily_snapshot": "daily",
    "decision_event": "decisions",
    "strategy_action": "strategy_actions",
    "portfolio_rule": "portfolio_rules",
    "risk_decision": "risk_decisions",
    "oms_intent": "oms_intents",
    "position_snapshot": "positions",
    "allocation_snapshot": "allocations",
    "portfolio_snapshot": "portfolio",
    "family_daily_snapshot": "family",
    "config_snapshot": "config_snapshots",
    "parameter_change": "config_changes",
    "deployment": "deployments",
    "resource_plan": "resource_plans",
    "market_data_subscription": "market_data",
    "reconciliation_event": "reconciliations",
    "session_closeout": "daily",
    "process_quality": "scores",
    "exit_movement": "exit_movements",
    "bot_error": "bot_errors",
    "error": "errors",
}

DIR_TO_EVENT_TYPE: dict[str, str] = {
    directory: event_type for event_type, directory in EVENT_DIRS.items()
}

# Legacy aliases whose historical directory names differ from the canonical
# event type names.
DIR_TO_EVENT_TYPE.update(
    {
        "trades": "trade",
        "missed": "missed_opportunity",
        "daily": "daily_snapshot",
        "snapshots": "market_snapshot",
        "config_changes": "parameter_change",
    }
)


EVENT_PRIORITIES: dict[str, int] = {
    "bot_error": 0,
    "error": 0,
    "risk_halt": 0,
    "reconciliation_alert": 0,
    "deployment": 1,
    "config_snapshot": 1,
    "daily_snapshot": 1,
    "session_closeout": 1,
    "trade": 2,
    "fill": 2,
    "missed_opportunity": 3,
    "order": 3,
    "portfolio_rule": 3,
    "risk_decision": 3,
    "parameter_change": 3,
    "oms_intent": 3,
    "decision_event": 4,
    "strategy_action": 4,
    "indicator_snapshot": 4,
    "filter_decision": 4,
    "market_snapshot": 4,
    "orderbook_context": 4,
    "position_snapshot": 4,
    "allocation_snapshot": 4,
    "portfolio_snapshot": 4,
    "resource_plan": 4,
    "market_data_subscription": 4,
    "reconciliation_event": 4,
    "process_quality": 4,
    "exit_movement": 4,
    "family_daily_snapshot": 4,
    "heartbeat": 5,
}


EVENT_SCOPES: dict[str, str] = {
    "deployment": "portfolio",
    "config_snapshot": "portfolio",
    "daily_snapshot": "portfolio",
    "session_closeout": "portfolio",
    "resource_plan": "portfolio",
    "market_data_subscription": "portfolio",
    "decision_event": "strategy",
    "strategy_action": "strategy",
    "filter_decision": "strategy",
    "indicator_snapshot": "strategy",
    "market_snapshot": "strategy",
    "orderbook_context": "strategy",
    "missed_opportunity": "strategy",
    "trade": "strategy",
    "portfolio_rule": "portfolio",
    "portfolio_snapshot": "portfolio",
    "family_daily_snapshot": "family",
    "risk_decision": "oms",
    "oms_intent": "oms",
    "order": "oms",
    "fill": "oms",
    "position_snapshot": "oms",
    "allocation_snapshot": "oms",
    "reconciliation_event": "oms",
    "heartbeat": "portfolio",
    "bot_error": "portfolio",
    "error": "portfolio",
}


EVENT_SCHEMA_VERSIONS: dict[str, str] = {
    event_type: f"{event_type}_v1" for event_type in EVENT_DIRS
}
EVENT_SCHEMA_VERSIONS.update(
    {
        "trade": "trade_event_v2",
        "missed_opportunity": "missed_opportunity_v2",
        "deployment": "deployment_event_v1",
        "config_snapshot": "config_snapshot_v1",
        "decision_event": "decision_event_v1",
        "strategy_action": "strategy_action_v1",
        "portfolio_rule": "portfolio_rule_v1",
        "risk_decision": "risk_decision_v1",
        "oms_intent": "oms_intent_v1",
        "order": "order_event_v2",
        "fill": "fill_event_v1",
    }
)


def event_dir(event_type: str) -> str:
    return EVENT_DIRS.get(str(event_type or ""), str(event_type or "events"))


def event_priority(event_type: str) -> int:
    return EVENT_PRIORITIES.get(str(event_type or ""), 4)


def event_scope(event_type: str, payload: Mapping[str, object] | None = None) -> str:
    raw_scope = (payload or {}).get("scope") if payload else None
    if raw_scope:
        return str(raw_scope)
    return EVENT_SCOPES.get(str(event_type or ""), "strategy")


def event_schema_version(event_type: str) -> str:
    return EVENT_SCHEMA_VERSIONS.get(str(event_type or ""), f"{event_type}_v1")
