from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .cache_keys import stable_signature
from .phase_state import _atomic_write_json, _utc_now_iso, load_phase_state

RUN_SPEC_FILENAME = "run_spec.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
OPTIMIZED_CONFIG_FILENAME = "optimized_config.json"
ROUND_DIAGNOSTICS_FILENAME = "round_final_diagnostics.txt"
ROUND_EVALUATION_FILENAME = "round_evaluation.txt"
PHASE_STATE_FILENAME = "phase_state.json"
MANIFEST_FILENAME = "rounds_manifest.json"
_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def canonicalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    raw = metrics or {}
    contract = raw.get("metric_contract") if isinstance(raw.get("metric_contract"), dict) else {}
    execution_contract = raw.get("execution_contract")
    if not isinstance(execution_contract, dict):
        execution_contract = contract.get("execution_contract") if isinstance(contract.get("execution_contract"), dict) else {}
    return {
        "total_trades": raw.get("total_trades", raw.get("trades")),
        "win_rate": raw.get("win_rate"),
        "profit_factor": raw.get("profit_factor", raw.get("pf")),
        "max_drawdown_pct": raw.get("max_drawdown_pct", raw.get("max_dd_pct")),
        "net_return_pct": raw.get("net_return_pct", raw.get("return_pct")),
        "net_return_pct_basis": raw.get("net_return_pct_basis"),
        "official_mtm_net_return_pct": raw.get("official_mtm_net_return_pct"),
        "official_metric_basis": raw.get("official_metric_basis"),
        "primary_promotion_metric": raw.get("primary_promotion_metric", contract.get("primary_promotion_metric")),
        "primary_promotion_value": raw.get("primary_promotion_value", contract.get("primary_promotion_value")),
        "primary_promotion_basis": raw.get("primary_promotion_basis", contract.get("primary_promotion_basis")),
        "official_replay_pass": raw.get("official_replay_pass", contract.get("official_replay_pass")),
        "audit_pass": raw.get("audit_pass", contract.get("audit_pass")),
        "audit_status": raw.get("audit_status", contract.get("audit_status")),
        "promotion_status": raw.get("promotion_status"),
        "promotion_requires_audit_pass": raw.get("promotion_requires_audit_pass", contract.get("promotion_requires_audit_pass")),
        "source_fingerprint": raw.get("source_fingerprint", raw.get("source_data_fingerprint", contract.get("source_fingerprint", execution_contract.get("source_fingerprint")))),
        "feature_manifest_hash": raw.get("feature_manifest_hash", contract.get("feature_manifest_hash", execution_contract.get("feature_manifest_hash"))),
        "candidate_snapshot_hash": raw.get("candidate_snapshot_hash", contract.get("candidate_snapshot_hash", execution_contract.get("candidate_snapshot_hash"))),
        "cost_policy_hash": raw.get("cost_policy_hash", _hash_contract_value(execution_contract.get("cost_policy"))),
        "fill_timing": raw.get("fill_timing", execution_contract.get("fill_timing")),
        "auction_mode": raw.get("auction_mode", execution_contract.get("auction_mode")),
        "capability_level": raw.get("capability_level", execution_contract.get("capability_level")),
        "same_bar_fill_count": raw.get("same_bar_fill_count"),
        "forced_replay_close_count": raw.get("forced_replay_close_count"),
        "rejected_order_count": raw.get("rejected_order_count"),
        "end_open_position_count": raw.get("end_open_position_count"),
        "sharpe_ratio": raw.get("sharpe_ratio", raw.get("sharpe")),
    }


def _hash_contract_value(value: Any) -> str:
    return "" if value in (None, "", {}, []) else stable_signature(value)


def _metrics_with_metadata(metrics: dict[str, Any] | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(metrics or {})
    for key, value in dict(metadata or {}).items():
        if key in {
            "promotion_status",
            "metric_contract",
            "execution_contract",
            "source_fingerprint",
            "source_data_fingerprint",
            "feature_manifest_hash",
            "candidate_snapshot_hash",
            "cost_policy_hash",
            "fill_timing",
            "auction_mode",
            "capability_level",
        } and value not in (None, ""):
            merged.setdefault(key, value)
    return merged


class RoundManager:
    def __init__(self, family: str, strategy: str, base_dir: Path | None = None):
        self.family = family
        self.strategy = strategy
        self.base_dir = Path(base_dir or Path("data/backtests/output"))
        self.strategy_dir = self.base_dir / strategy
        self.manifest_path = self.strategy_dir / MANIFEST_FILENAME

    def round_path(self, round_num: int) -> Path:
        return self.strategy_dir / f"round_{round_num}"

    def get_round_dir(self, round_num: int) -> Path:
        path = self.round_path(round_num)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def phase_state_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / PHASE_STATE_FILENAME

    def diagnostics_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_DIAGNOSTICS_FILENAME

    def evaluation_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_EVALUATION_FILENAME

    def run_summary_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / RUN_SUMMARY_FILENAME

    def optimized_config_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / OPTIMIZED_CONFIG_FILENAME

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"family": self.family, "strategy": self.strategy, "rounds": []}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data.setdefault("family", self.family)
        data.setdefault("strategy", self.strategy)
        data.setdefault("rounds", [])
        return data

    def get_latest_round(self) -> int:
        latest = 0
        for item in self.load_manifest().get("rounds", []):
            if item.get("archived"):
                continue
            latest = max(latest, int(item.get("round", 0) or 0))
        return latest

    def get_archived_rounds(self) -> set[int]:
        return {
            int(item.get("round", 0) or 0)
            for item in self.load_manifest().get("rounds", [])
            if item.get("archived")
        }

    def previous_active_round(self, current_round: int) -> int:
        archived_rounds = self.get_archived_rounds()
        previous = int(current_round) - 1
        while previous in archived_rounds:
            previous -= 1
        return previous

    def resolve_round(self, requested_round: int | None, *, for_write: bool, expected_phases: int | None = None) -> tuple[int, Path]:
        latest = self.get_latest_round()
        if requested_round is not None:
            if requested_round < 1:
                raise ValueError("Round numbers must be positive")
            if for_write:
                return requested_round, self.get_round_dir(requested_round)
            path = self.round_path(requested_round)
            if not path.exists():
                raise FileNotFoundError(path)
            return requested_round, path
        if not for_write:
            if latest < 1:
                raise FileNotFoundError(f"No rounds exist under {self.strategy_dir}")
            return latest, self.round_path(latest)
        archived_rounds = self.get_archived_rounds()
        if latest == 0:
            return self._next_writable_round(1, archived_rounds)
        latest_dir = self.round_path(latest)
        if self.run_summary_path(latest_dir).exists():
            return self._next_writable_round(latest + 1, archived_rounds)
        state_path = self.phase_state_path(latest_dir)
        if state_path.exists() and expected_phases is not None:
            state = load_phase_state(state_path)
            if len(state.completed_phases) < expected_phases:
                return latest, self.get_round_dir(latest)
        return latest, self.get_round_dir(latest)

    def get_previous_mutations(
        self,
        current_round: int | None = None,
        *,
        expected_execution_contract: dict[str, Any] | None = None,
        allow_incompatible_baseline: bool = False,
    ) -> dict[str, Any]:
        previous = self.get_latest_round() if current_round is None else self.previous_active_round(current_round)
        if previous < 1:
            return {}
        data = self.load_optimized_config(previous)
        if not data:
            return {}
        if expected_execution_contract:
            actual = data.get("execution_contract") if isinstance(data, dict) else {}
            mismatches = _execution_contract_mismatches(expected_execution_contract, actual if isinstance(actual, dict) else {})
            if mismatches and not allow_incompatible_baseline:
                details = ", ".join(f"{key}: expected={expected!r} actual={actual!r}" for key, expected, actual in mismatches[:8])
                raise ValueError(
                    f"Previous optimized config for {self.strategy} round {previous} was produced under an incompatible execution contract: {details}"
                )
        return dict(data.get("mutations", data)) if isinstance(data, dict) else {}

    def previous_optimized_config_path(self, current_round: int) -> Path:
        previous = self.previous_active_round(current_round)
        return self.optimized_config_path(self.round_path(previous)) if previous >= 1 else Path()

    def load_optimized_config(self, round_num: int) -> dict[str, Any]:
        path = self.optimized_config_path(self.round_path(round_num))
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def write_run_spec(
        self,
        round_dir: Path,
        round_num: int,
        strategy_name: str,
        *,
        description: str = "",
        baseline_mutations: dict[str, Any] | None = None,
        baseline_source: str | Path | None = None,
        baseline_metadata: dict[str, Any] | None = None,
        execution_context: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        path = Path(round_dir) / RUN_SPEC_FILENAME
        if path.exists() and not overwrite:
            return path
        _atomic_write_json(
            {
                "family": self.family,
                "strategy": self.strategy,
                "strategy_name": strategy_name,
                "round": round_num,
                "description": description,
                "generated_at_utc": _utc_now_iso(),
                "baseline_source": str(baseline_source) if baseline_source else None,
                "baseline_mutations": dict(baseline_mutations or {}),
                "baseline_metadata": dict(baseline_metadata or {}),
                "execution_context": execution_context or {},
            },
            path,
        )
        return path

    def write_run_summary(
        self,
        round_dir: Path,
        cumulative_mutations: dict[str, Any],
        final_metrics: dict[str, Any] | None,
        completed_phases: list[int],
        *,
        round_num: int | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Path:
        resolved_round = round_num if round_num is not None else self._round_num_from_dir(round_dir)
        path = self.run_summary_path(round_dir)
        metadata = dict(artifact_metadata or {})
        summary_metrics = _metrics_with_metadata(final_metrics, metadata)
        _atomic_write_json(
            {
                "family": self.family,
                "strategy": self.strategy,
                "round": resolved_round,
                "generated_at_utc": _utc_now_iso(),
                "completed_phases": completed_phases,
                "cumulative_mutations": cumulative_mutations,
                "headline_metrics": canonicalize_metrics(summary_metrics),
                "final_metrics": final_metrics or {},
                **metadata,
            },
            path,
        )
        return path

    def write_optimized_config(
        self,
        round_dir: Path,
        cumulative_mutations: dict[str, Any],
        *,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.optimized_config_path(round_dir)
        metadata = dict(artifact_metadata or {})
        _atomic_write_json(
            {
                "mutations": dict(cumulative_mutations),
                "generated_at_utc": _utc_now_iso(),
                **metadata,
            },
            path,
        )
        return path

    def append_to_manifest(
        self,
        round_num: int,
        cumulative_mutations: dict[str, Any],
        final_metrics: dict[str, Any] | None,
        *,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Path:
        manifest = self.load_manifest()
        metadata = dict(artifact_metadata or {})
        raw_metrics = _metrics_with_metadata(final_metrics, metadata)
        contract = raw_metrics.get("metric_contract") if isinstance(raw_metrics.get("metric_contract"), dict) else {}
        entry = {
            "round": round_num,
            "timestamp": _utc_now_iso(),
            "mutations_count": len(cumulative_mutations),
            "mutations": dict(cumulative_mutations),
            **canonicalize_metrics(raw_metrics),
        }
        for key in ("promotion_status", "artifact_promotion_policy"):
            if metadata.get(key) not in (None, ""):
                entry[key] = metadata[key]
        metadata_contract = metadata.get("metric_contract")
        if metadata_contract:
            entry["metric_contract"] = metadata_contract
        elif contract:
            entry["metric_contract"] = contract
        execution_contract = raw_metrics.get("execution_contract", contract.get("execution_contract"))
        if metadata.get("execution_contract"):
            entry["execution_contract"] = metadata["execution_contract"]
        elif execution_contract:
            entry["execution_contract"] = execution_contract
        rounds = manifest.setdefault("rounds", [])
        rounds[:] = [item for item in rounds if int(item.get("round", 0) or 0) != round_num]
        rounds.append(entry)
        rounds.sort(key=lambda item: int(item.get("round", 0) or 0))
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest, self.manifest_path)
        return self.manifest_path

    @staticmethod
    def _round_num_from_dir(round_dir: Path) -> int:
        match = _ROUND_DIR_RE.match(Path(round_dir).name)
        if not match:
            raise ValueError(f"Could not infer round number from {round_dir}")
        return int(match.group(1))

    def _next_writable_round(self, start: int, archived_rounds: set[int]) -> tuple[int, Path]:
        round_num = int(start)
        while round_num in archived_rounds:
            round_num += 1
        return round_num, self.get_round_dir(round_num)


def _execution_contract_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    if not expected:
        return []
    strict_keys = (
        "strategy",
        "phase_framework_version",
        "strategy_core_version",
        "source_fingerprint",
        "feature_manifest_hash",
        "candidate_snapshot_hash",
        "date_window",
        "initial_equity",
        "cost_policy",
        "fill_timing",
        "auction_mode",
        "capability_level",
        "replay_mode",
        "primary_promotion_metric",
        "primary_promotion_basis",
    )
    mismatches: list[tuple[str, Any, Any]] = []
    for key in strict_keys:
        expected_value = expected.get(key)
        if expected_value in (None, "", {}, []):
            continue
        actual_value = actual.get(key)
        if not _contract_value_matches(expected_value, actual_value):
            mismatches.append((key, expected_value, actual_value))
    return mismatches


def _contract_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(_contract_value_matches(value, actual.get(key)) for key, value in expected.items())
    return actual == expected
