from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .offline_replay import is_engine_replay_manifest, load_offline_replay_manifest
from .kis_resource_plan import resource_plan_hash
from .session_capture import REQUIRED_DIRS, REQUIRED_EXPECTED_HASH_GROUPS, REQUIRED_JSONL, missing_artifact_evidence, session_hashes

MISMATCH_CLASSES = (
    "artifact_mismatch",
    "market_data_mismatch",
    "strategy_decision_mismatch",
    "action_serialization_mismatch",
    "portfolio_policy_mismatch",
    "oms_normalization_mismatch",
    "broker_fill_mismatch",
    "state_hydration_mismatch",
    "end_position_mismatch",
)
REQUIRED_SINGLE_FILES = ("session_manifest.json", "market_bars_5m.parquet", "end_of_day_positions.json")
OFFLINE_REPLAY_DIR = "offline_replay"
OFFLINE_REPLAY_BEHAVIOR_HASH_KEYS = (
    "decision_stream",
    "strategy_actions",
    "portfolio_arbitration",
    "oms_intents",
    "order_events",
    "fill_events",
    "trade_outcomes",
    "state_snapshots",
    "end_of_day_positions",
)
OFFLINE_REPLAY_INPUT_HASH_KEYS = (
    "artifact_generation",
    "subscription_events",
    "daily_snapshots_manifest",
    "olr_stage1_snapshots_manifest",
    "olr_final_snapshots_manifest",
    "strategy_configs_manifest",
    "kis_resource_plan",
    "market_bars_5m",
    "runtime_events",
)
OFFLINE_REPLAY_HASH_KEYS = OFFLINE_REPLAY_INPUT_HASH_KEYS + OFFLINE_REPLAY_BEHAVIOR_HASH_KEYS
OFFLINE_REPLAY_REQUIRED_FILES = {
    "decision_stream": "decision_stream.jsonl",
    "strategy_actions": "strategy_actions.jsonl",
    "portfolio_arbitration": "portfolio_arbitration.jsonl",
    "oms_intents": "oms_intents.jsonl",
    "order_events": "order_events.jsonl",
    "fill_events": "fill_events.jsonl",
    "trade_outcomes": "trade_outcomes.jsonl",
    "state_snapshots": "state_snapshots.jsonl",
    "end_of_day_positions": "end_of_day_positions.json",
}


def replay_paper_session(session_root: str | Path) -> dict[str, Any]:
    root = Path(session_root)
    if not root.exists():
        raise FileNotFoundError(root)
    manifest_path = root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    hashes = session_hashes(root)
    missing_files = _missing_required_files(root)
    resource_plan_required = manifest.get("kis_resource_plan_required") is True or str(manifest.get("mode") or "").lower() in {"paper", "live"}
    resource_plan_failures = _resource_plan_contract_failures(root, manifest, required=resource_plan_required)
    resource_plan_missing = "kis_resource_plan_missing" in resource_plan_failures
    missing_dirs = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    missing_artifacts = missing_artifact_evidence(root, manifest.get("strategy_ids"))
    expected = dict(manifest.get("expected_hashes") or {})
    hash_contract_status = str(manifest.get("hash_contract_status") or "")
    expected_hash_groups = tuple(str(item) for item in (manifest.get("expected_hash_groups") or ()))
    expected_hashes_complete = manifest.get("expected_hashes_complete") is True
    required_hash_groups = tuple(REQUIRED_EXPECTED_HASH_GROUPS)
    expected_keys = set(expected)
    required_keys = set(required_hash_groups)
    missing_expected_hash_groups = sorted(required_keys - expected_keys)
    unknown_expected_hash_groups = sorted(expected_keys - required_keys)
    missing_actual_hash_groups = sorted(key for key in required_hash_groups if key not in hashes)
    expected_group_set_valid = set(expected_hash_groups) == required_keys and len(expected_hash_groups) == len(required_hash_groups)
    mismatches = []
    for key, expected_hash in expected.items():
        actual_hash = hashes.get(key)
        if actual_hash != expected_hash:
            mismatches.append({"class": _classify_hash_key(key), "key": key, "expected": expected_hash, "actual": actual_hash})
    hash_contract_available = (
        bool(expected)
        and hash_contract_status == "sealed"
        and expected_hashes_complete
        and expected_group_set_valid
        and not missing_expected_hash_groups
        and not unknown_expected_hash_groups
        and not missing_actual_hash_groups
    )
    hash_contract_passed = hash_contract_available and not mismatches
    session_bundle_complete = not missing_files and not missing_dirs and not missing_artifacts and not resource_plan_failures
    offline_rebuild = _offline_replay_comparison(root, hashes)
    mismatches.extend(offline_rebuild["mismatches"])
    offline_rebuild_implemented = bool(offline_rebuild["implemented"])
    offline_rebuild_passed = offline_rebuild_implemented and not offline_rebuild["mismatches"]
    behavior_parity_passed = offline_rebuild_passed and hash_contract_passed and session_bundle_complete
    promotion_blockers = []
    if not offline_rebuild_implemented:
        promotion_blockers.append("offline_rebuild_not_implemented")
    elif not offline_rebuild_passed:
        promotion_blockers.append("offline_rebuild_mismatch")
    if not session_bundle_complete:
        promotion_blockers.append("session_bundle_incomplete")
    if resource_plan_missing:
        promotion_blockers.append("kis_resource_plan_missing")
    if resource_plan_failures and not resource_plan_missing:
        promotion_blockers.append("kis_resource_plan_invalid")
    if not expected:
        promotion_blockers.append("hash_contract_missing")
    if expected and hash_contract_status != "sealed":
        promotion_blockers.append("hash_contract_unsealed")
    if expected and (
        not expected_hashes_complete
        or not expected_group_set_valid
        or missing_expected_hash_groups
        or unknown_expected_hash_groups
        or missing_actual_hash_groups
    ):
        promotion_blockers.append("hash_contract_incomplete")
    if expected and mismatches:
        promotion_blockers.append("hash_contract_mismatch")
    paper_gate_status = (
        "passed"
        if behavior_parity_passed
        else ("offline_replay_built_mismatch" if offline_rebuild_implemented else "blocked_until_offline_rebuild")
    )
    report = {
        "session": str(root),
        "trade_date": manifest.get("trade_date"),
        "replay_mode": offline_rebuild["mode"],
        "offline_rebuild_implemented": offline_rebuild_implemented,
        "offline_rebuild_status": offline_rebuild["status"],
        "offline_rebuild_source": offline_rebuild["source"],
        "offline_rebuild_hashes": offline_rebuild["hashes"],
        "paper_gate_status": paper_gate_status,
        "promotion_blockers": promotion_blockers,
        "session_bundle_complete": session_bundle_complete,
        "missing_required_files": missing_files,
        "missing_required_dirs": missing_dirs,
        "missing_artifact_evidence": missing_artifacts,
        "resource_plan_failures": resource_plan_failures,
        "hashes": hashes,
        "hash_contract_available": hash_contract_available,
        "hash_contract_status": hash_contract_status,
        "hash_contract_expected_hashes_complete": expected_hashes_complete,
        "hash_contract_expected_groups": list(expected_hash_groups),
        "hash_contract_required_groups": list(required_hash_groups),
        "hash_contract_missing_expected_groups": missing_expected_hash_groups,
        "hash_contract_unknown_expected_groups": unknown_expected_hash_groups,
        "hash_contract_missing_actual_groups": missing_actual_hash_groups,
        "mismatches": mismatches,
        "mismatch_counts": {name: sum(1 for item in mismatches if item["class"] == name) for name in MISMATCH_CLASSES},
        "session_metrics": _session_metrics(root, manifest, missing_files=missing_files),
        "hash_contract_passed": hash_contract_passed,
        "behavior_parity_passed": behavior_parity_passed,
        "paper_gate_passed": behavior_parity_passed,
    }
    (root / "parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


def summarize_paper_parity(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    reports = []
    for path in sorted(base.glob("*/parity_report.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    mismatch_totals = {name: 0 for name in MISMATCH_CLASSES}
    metric_totals = {
        "rejected_orders": 0,
        "stale_bar_events": 0,
        "missing_bars": 0,
        "unexpected_end_positions": 0,
        "strategy_trade_attempts": 0,
    }
    portfolio_decisions = {"accepted": 0, "blocked": 0, "resized": 0, "deferred": 0}
    for report in reports:
        for name, count in dict(report.get("mismatch_counts") or {}).items():
            mismatch_totals[name] = mismatch_totals.get(name, 0) + int(count)
        metrics = dict(report.get("session_metrics") or {})
        for name in metric_totals:
            metric_totals[name] += int(metrics.get(name) or 0)
        for decision, count in dict(metrics.get("portfolio_decisions") or {}).items():
            portfolio_decisions[decision] = portfolio_decisions.get(decision, 0) + int(count)
    summary = {
        "root": str(base),
        "sessions_analyzed": len(reports),
        "sessions_passing": sum(1 for report in reports if report.get("behavior_parity_passed") is True),
        "sessions_with_hash_contract": sum(1 for report in reports if report.get("hash_contract_available") is True),
        "sessions_hash_contract_passing": sum(1 for report in reports if report.get("hash_contract_passed") is True),
        "sessions_paper_gate_passing": sum(1 for report in reports if report.get("paper_gate_passed") is True),
        "sessions_with_complete_bundles": sum(1 for report in reports if report.get("session_bundle_complete") is True),
        "mismatch_totals": mismatch_totals,
        **metric_totals,
        "portfolio_decisions": portfolio_decisions,
        "reports": [report.get("session") for report in reports],
    }
    (base / "paper_parity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def _classify_hash_key(key: str) -> str:
    if "artifact" in key or "snapshot" in key:
        return "artifact_mismatch"
    if "market_bars" in key:
        return "market_data_mismatch"
    if "decision" in key:
        return "strategy_decision_mismatch"
    if "action" in key:
        return "action_serialization_mismatch"
    if "portfolio" in key:
        return "portfolio_policy_mismatch"
    if "oms" in key:
        return "oms_normalization_mismatch"
    if "fill" in key:
        return "broker_fill_mismatch"
    if "state" in key:
        return "state_hydration_mismatch"
    if "position" in key:
        return "end_position_mismatch"
    return "strategy_decision_mismatch"


def _offline_replay_comparison(root: Path, live_hashes: dict[str, str]) -> dict[str, Any]:
    offline_root = root / OFFLINE_REPLAY_DIR
    if not offline_root.exists():
        return {
            "implemented": False,
            "mode": "hash_contract_only",
            "status": "missing_offline_replay",
            "source": "",
            "hashes": {},
            "mismatches": [],
        }
    replay_manifest = load_offline_replay_manifest(offline_root)
    if not is_engine_replay_manifest(replay_manifest):
        status = "missing_engine_replay_manifest" if not replay_manifest else "invalid_engine_replay_manifest"
        return {
            "implemented": False,
            "mode": "external_offline_stream_contract",
            "status": status,
            "source": str(offline_root),
            "hashes": {},
            "mismatches": [],
        }
    offline_hashes = session_hashes(offline_root)
    missing = {
        key
        for key, filename in OFFLINE_REPLAY_REQUIRED_FILES.items()
        if not (offline_root / filename).is_file()
    }
    mismatches = []
    if replay_manifest.get("driver_replay") is not True:
        mismatches.append(
            {
                "class": "state_hydration_mismatch",
                "key": "offline_replay.driver_replay",
                "expected": True,
                "actual": replay_manifest.get("driver_replay"),
            }
        )
    fill_status = str(replay_manifest.get("fill_replay_status") or "")
    if fill_status not in {"not_applicable_no_fill_events", "replayed"}:
        mismatches.append(
            {
                "class": "broker_fill_mismatch",
                "key": "offline_replay.fill_replay_status",
                "expected": "replayed_or_absent",
                "actual": fill_status or "missing",
            }
        )
    timer_count = _count_timer_event_inputs(root)
    timer_status = str(replay_manifest.get("timer_replay_status") or "")
    expected_timer_status = "replayed" if timer_count else "not_applicable_no_timer_events"
    if int(replay_manifest.get("timer_event_count") or 0) != timer_count:
        mismatches.append(
            {
                "class": "strategy_decision_mismatch",
                "key": "offline_replay.timer_event_count",
                "expected": timer_count,
                "actual": replay_manifest.get("timer_event_count"),
            }
        )
    if timer_status != expected_timer_status:
        mismatches.append(
            {
                "class": "strategy_decision_mismatch",
                "key": "offline_replay.timer_replay_status",
                "expected": expected_timer_status,
                "actual": timer_status or "missing",
            }
        )
    order_status = str(replay_manifest.get("order_event_replay_status") or "replayed_or_absent")
    if order_status != "replayed_or_absent":
        mismatches.append(
            {
                "class": "broker_fill_mismatch",
                "key": "offline_replay.order_event_replay_status",
                "expected": "replayed_or_absent",
                "actual": order_status,
            }
        )
    input_hashes = dict(replay_manifest.get("input_hashes") or {})
    for key in OFFLINE_REPLAY_INPUT_HASH_KEYS:
        expected = live_hashes.get(key)
        actual = input_hashes.get(key)
        if expected != actual:
            report_key = "offline_replay.market_bars_5m_input_hash" if key == "market_bars_5m" else f"offline_replay.{key}"
            mismatches.append({"class": _classify_hash_key(key), "key": report_key, "expected": expected, "actual": actual})
    for key in OFFLINE_REPLAY_BEHAVIOR_HASH_KEYS:
        if key in missing:
            mismatches.append({"class": _classify_hash_key(key), "key": f"offline_replay.{key}", "expected": live_hashes.get(key), "actual": None})
        elif live_hashes.get(key) != offline_hashes.get(key):
            mismatches.append({"class": _classify_hash_key(key), "key": f"offline_replay.{key}", "expected": live_hashes.get(key), "actual": offline_hashes.get(key)})
    return {
        "implemented": True,
        "mode": "offline_replay_built" if mismatches else "paper_gate_passed",
        "status": "engine_replay_compared",
        "source": str(offline_root),
        "hashes": {
            **{f"{key}_input": input_hashes.get(key) for key in OFFLINE_REPLAY_INPUT_HASH_KEYS},
            **{key: offline_hashes.get(key) for key in OFFLINE_REPLAY_BEHAVIOR_HASH_KEYS},
        },
        "mismatches": mismatches,
    }


def _missing_required_files(root: Path) -> list[str]:
    missing = [name for name in REQUIRED_SINGLE_FILES if not (root / name).is_file()]
    missing.extend(filename for filename in REQUIRED_JSONL if not (root / filename).is_file())
    return sorted(missing)


def _resource_plan_contract_failures(root: Path, manifest: dict[str, Any], *, required: bool) -> list[str]:
    path = root / "kis_resource_plan.json"
    if not path.is_file():
        return ["kis_resource_plan_missing"] if required else []
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return ["kis_resource_plan_invalid_json"]
    failures: list[str] = []
    declared_hash = str(payload.get("plan_hash") or "")
    actual_hash = resource_plan_hash(payload)
    manifest_hash = str(manifest.get("kis_resource_plan_hash") or "")
    if not declared_hash:
        failures.append("kis_resource_plan_hash_missing")
    elif declared_hash != actual_hash:
        failures.append("kis_resource_plan_hash_mismatch")
    if required and not manifest_hash:
        failures.append("kis_resource_plan_manifest_hash_missing")
    elif manifest_hash and declared_hash and manifest_hash != declared_hash:
        failures.append("kis_resource_plan_manifest_hash_mismatch")
    if payload.get("passed") is not True:
        failures.append("kis_resource_plan_not_passed")
    return failures


def _session_metrics(root: Path, manifest: dict[str, Any], *, missing_files: list[str]) -> dict[str, Any]:
    metrics = dict(manifest.get("session_metrics") or {})
    portfolio_counts = {"accepted": 0, "blocked": 0, "resized": 0, "deferred": 0}
    for row in _read_jsonl(root / "portfolio_arbitration.jsonl"):
        decision = str(row.get("decision") or "").strip().lower()
        if decision in portfolio_counts:
            portfolio_counts[decision] += 1
    manifest_portfolio_counts = dict(metrics.get("portfolio_decisions") or {})
    return {
        "rejected_orders": _metric_int(metrics, "rejected_orders", _count_rejected_orders(root)),
        "stale_bar_events": _metric_int(metrics, "stale_bar_events", _count_stale_bar_events(root)),
        "missing_bars": _metric_int(metrics, "missing_bars", int("market_bars_5m.parquet" in missing_files)),
        "unexpected_end_positions": _metric_int(metrics, "unexpected_end_positions", _count_unexpected_end_positions(root)),
        "strategy_trade_attempts": _metric_int(metrics, "strategy_trade_attempts", _count_strategy_trade_attempts(root)),
        "portfolio_decisions": {
            key: int(manifest_portfolio_counts[key]) if key in manifest_portfolio_counts else portfolio_counts[key]
            for key in portfolio_counts
        },
    }


def _metric_int(metrics: dict[str, Any], key: str, fallback: int) -> int:
    return int(metrics[key]) if key in metrics else int(fallback)


def _count_rejected_orders(root: Path) -> int:
    return sum(
        1
        for row in _read_jsonl(root / "order_events.jsonl")
        if str(row.get("status") or row.get("event_type") or row.get("reason_code") or "").lower().startswith("reject")
        or bool(row.get("rejected"))
    )


def _count_stale_bar_events(root: Path) -> int:
    return sum(
        1
        for row in _read_jsonl(root / "decision_stream.jsonl")
        if bool(row.get("stale_bar"))
        or str(row.get("decision_code") or row.get("reason_code") or "").lower() in {"stale_bar", "stale_market_data"}
    )


def _count_timer_event_inputs(root: Path) -> int:
    return sum(
        1
        for row in _read_jsonl(root / "decision_stream.jsonl")
        if row.get("record_type") == "runtime_event_input" and row.get("event_type") == "timer"
    )


def _count_unexpected_end_positions(root: Path) -> int:
    path = root / "end_of_day_positions.json"
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if "unexpected_positions" in payload:
        return len(payload.get("unexpected_positions") or [])
    return sum(1 for row in payload.get("positions") or [] if row.get("unexpected") is True)


def _count_strategy_trade_attempts(root: Path) -> int:
    noops = {"", "noop", "hold", "none"}
    return sum(
        1
        for row in _read_jsonl(root / "strategy_actions.jsonl")
        if str(row.get("action") or row.get("action_type") or row.get("type") or "").strip().lower() not in noops
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
