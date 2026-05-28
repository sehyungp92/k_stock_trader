from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backtests.strategies.common.plugin_base import build_execution_contract

from .evaluation import build_end_of_round_report
from .greedy_optimizer import run_greedy
from .phase_analyzer import analyze_phase
from .phase_gates import evaluate_gate
from .phase_logging import PhaseLogger
from .phase_state import PhaseState, _atomic_write_json, _utc_now_iso, load_phase_state, save_phase_state
from .plugin import PhaseSpec, StrategyPlugin
from .round_manager import RoundManager
from .types import GateCriterion, GateResult, GreedyResult, PhaseAnalysis


class PhaseRunner:
    def __init__(
        self,
        plugin: StrategyPlugin,
        output_dir: Path,
        round_name: str = "",
        *,
        max_rounds: int | None = None,
        min_delta: float = 0.001,
        max_retries: int = 2,
        max_diagnostic_retries: int = 1,
        round_manager: RoundManager | None = None,
        round_num: int | None = None,
    ):
        self.plugin = plugin
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_name = round_name
        self.max_rounds = max_rounds
        self.min_delta = min_delta
        self.max_retries = max_retries
        self.max_diagnostic_retries = max_diagnostic_retries
        self.round_manager = round_manager
        self.round_num = round_num
        self.state_path = (
            self.round_manager.phase_state_path(self.output_dir)
            if self.round_manager else self.output_dir / "phase_state.json"
        )
        self.phase_logger = PhaseLogger(self.output_dir, round_name=round_name)

    def load_state(self) -> PhaseState:
        state = load_phase_state(self.state_path)
        if not state.cumulative_mutations and not state.phase_results:
            state.cumulative_mutations = self._initial_baseline_mutations()
        else:
            state.cumulative_mutations = self._canonicalize_mutations(state.cumulative_mutations)
        if self.round_name and not state.round_name:
            state.round_name = self.round_name
        return state

    def run_all_phases(self, start_phase: int | None = None) -> PhaseState:
        initial_baseline = self._initial_baseline_mutations()
        baseline_integrity = self._validate_initial_baseline(initial_baseline)
        state = self.load_state()
        save_phase_state(state, self.state_path)
        if self.round_manager and self.round_num is not None:
            execution_context = _plugin_execution_context(self.plugin)
            if baseline_integrity:
                execution_context["baseline_integrity"] = baseline_integrity
            self.round_manager.write_run_spec(
                self.output_dir,
                self.round_num,
                self.plugin.name,
                description=self.round_name or f"Round {self.round_num}",
                baseline_mutations=dict(initial_baseline),
                baseline_source=self._initial_baseline_source(),
                baseline_metadata=baseline_integrity,
                execution_context=execution_context,
                overwrite=True,
            )
        start = start_phase or (max(state.completed_phases) + 1 if state.completed_phases else 1)
        ran_final_phase = False
        try:
            for phase in range(start, self.plugin.num_phases + 1):
                state = self.run_phase(phase, state)
                ran_final_phase = phase == self.plugin.num_phases
            if not ran_final_phase and _round_is_complete(state, self.plugin.num_phases):
                self.run_end_of_round(state)
        finally:
            close_pool = getattr(self.plugin, "close_pool", None)
            if callable(close_pool):
                close_pool()
        return state

    def run_phase(self, phase: int, state: PhaseState | None = None) -> PhaseState:
        state = self._prepare_state_for_phase(state or self.load_state(), phase)
        state.cumulative_mutations = self._canonicalize_mutations(state.cumulative_mutations)
        base_mutations = dict(state.cumulative_mutations)
        state.start_phase(phase)
        save_phase_state(state, self.state_path)
        spec = self.plugin.get_phase_spec(phase, state)
        candidates = _dedupe_experiments(spec.candidates)
        scoring_weights = spec.scoring_weights
        greedy_result = None
        metrics = None
        gate_result = None
        force_enhanced_diagnostics = False
        log = self.phase_logger.get_phase_logger(phase)
        self.phase_logger.log_activity(phase, "phase_start", {"focus": spec.focus, "candidate_count": len(candidates)})
        self._progress(state, phase, "phase_started", spec.focus, len(candidates))

        while True:
            if greedy_result is None:
                evaluator = self.plugin.create_evaluate_batch(
                    phase,
                    base_mutations,
                    scoring_weights=scoring_weights,
                    hard_rejects=spec.hard_rejects,
                )
                setter = getattr(evaluator, "set_progress_callback", None)
                if callable(setter):
                    setter(lambda payload: self._progress(state, phase, "greedy_running", spec.focus, len(candidates), {"greedy_progress": payload}))
                checkpoint_path = self.output_dir / f"phase_{phase}_greedy_checkpoint.json"
                greedy_result = run_greedy(
                    candidates,
                    base_mutations,
                    evaluator,
                    max_rounds=self.max_rounds or spec.max_rounds or len(candidates),
                    min_delta=self.min_delta,
                    prune_threshold=spec.prune_threshold if spec.prune_threshold is not None else 0.05,
                    reject_streak_limit=spec.reject_streak_limit if spec.reject_streak_limit is not None else 1,
                    checkpoint_path=checkpoint_path,
                    checkpoint_context={"phase": phase, "scoring_weights": scoring_weights or {}, "hard_rejects": spec.hard_rejects},
                    logger=log,
                )
                self.phase_logger.save_phase_output(phase, "greedy_raw", _to_dict(greedy_result))
                greedy_result.final_mutations = self._canonicalize_mutations(greedy_result.final_mutations)
                metrics = _apply_phase_metric_contract(
                    self.plugin.compute_final_metrics(greedy_result.final_mutations),
                    spec,
                    self.plugin,
                )
                greedy_result.final_metrics = dict(metrics or {})
                self.phase_logger.save_phase_output(phase, "greedy", _to_dict(greedy_result))
                extra_criteria = _plugin_phase_acceptance_criteria(self.plugin, phase, base_mutations, metrics, greedy_result)
                gate_result = evaluate_gate([*spec.gate_criteria_fn(metrics), *extra_criteria, *_metric_contract_criteria(metrics)], greedy_result)
                state.record_gate(phase, _gate_to_dict(gate_result))
                save_phase_state(state, self.state_path)
                self.phase_logger.log_activity(phase, "gate_check", {"passed": gate_result.passed, "failure_category": gate_result.failure_category})
                self._progress(state, phase, "gate_passed" if gate_result.passed else "gate_failed", spec.focus, len(candidates))

            assert greedy_result is not None and metrics is not None and gate_result is not None
            diagnostics = (
                self.plugin.run_enhanced_diagnostics(phase, state, metrics, greedy_result)
                if force_enhanced_diagnostics
                else self.plugin.run_phase_diagnostics(phase, state, metrics, greedy_result)
            )
            self.phase_logger.save_phase_output(phase, "diagnostics_enhanced" if force_enhanced_diagnostics else "diagnostics", diagnostics)
            analysis = analyze_phase(
                phase,
                greedy_result,
                metrics,
                state,
                gate_result,
                ultimate_targets=self.plugin.ultimate_targets,
                policy=spec.analysis_policy,
                current_weights=scoring_weights,
                max_scoring_retries=self.max_retries,
                max_diagnostic_retries=self.max_diagnostic_retries,
            )
            self.phase_logger.save_phase_output(phase, "analysis", _analysis_to_dict(analysis))
            self.phase_logger.save_phase_output(phase, "analysis", analysis.report)
            self.phase_logger.log_activity(phase, "analysis_complete", {"recommendation": analysis.recommendation, "scoring_assessment": analysis.scoring_assessment})

            if analysis.recommendation == "improve_scoring" and state.scoring_retries.get(phase, 0) < self.max_retries:
                state.increment_retry(phase)
                retry = state.increment_scoring_retry(phase)
                scoring_weights = analysis.scoring_weight_overrides or scoring_weights
                if scoring_weights:
                    _atomic_write_json(scoring_weights, self.output_dir / f"phase_{phase}_score_spec_v{retry}.json")
                if analysis.suggested_experiments:
                    candidates = _dedupe_experiments([*candidates, *analysis.suggested_experiments])
                greedy_result = None
                metrics = None
                gate_result = None
                save_phase_state(state, self.state_path)
                self.phase_logger.log_activity(phase, "decision_improve_scoring", {"retry": retry})
                continue

            if analysis.recommendation == "improve_diagnostics" and state.diagnostic_retries.get(phase, 0) < self.max_diagnostic_retries:
                state.increment_retry(phase)
                retry = state.increment_diagnostic_retry(phase)
                force_enhanced_diagnostics = True
                save_phase_state(state, self.state_path)
                self.phase_logger.log_activity(phase, "decision_improve_diagnostics", {"retry": retry})
                continue

            applied = gate_result.passed
            adopted_mutations = self._canonicalize_mutations(greedy_result.final_mutations if applied else base_mutations)
            adopted_metrics = metrics if applied else _apply_phase_metric_contract(
                self.plugin.compute_final_metrics(base_mutations),
                spec,
                self.plugin,
            )
            new_mutations = {
                key: value
                for key, value in adopted_mutations.items()
                if base_mutations.get(key, object()) != value
            }
            result = {
                "focus": spec.focus,
                "base_mutations": base_mutations,
                "final_mutations": adopted_mutations,
                "base_score": greedy_result.base_score,
                "final_score": greedy_result.final_score if applied else greedy_result.base_score,
                "kept_features": list(greedy_result.kept_features if applied else []),
                "rounds": [_to_dict(item) for item in greedy_result.rounds],
                "final_metrics": adopted_metrics,
                "total_candidates": greedy_result.total_candidates,
                "accepted_count": greedy_result.accepted_count if applied else 0,
                "elapsed_seconds": greedy_result.elapsed_seconds,
                "suggested_experiments": [_to_dict(item) for item in analysis.suggested_experiments],
                "analysis": _analysis_to_dict(analysis),
                "new_mutations": new_mutations,
                "applied_phase_mutations": applied,
                "adoption_reason": "gate_passed" if applied else "gate_failed",
            }
            state.advance_phase(phase, new_mutations, result)
            state.record_gate(phase, _gate_to_dict(gate_result))
            save_phase_state(state, self.state_path)
            self.phase_logger.log_activity(phase, "decision_advance", {"gate_passed": gate_result.passed, "new_mutation_count": len(new_mutations)})
            self._progress(state, phase, "completed", spec.focus, len(candidates))
            if phase == self.plugin.num_phases:
                self.run_end_of_round(state)
            return state

    def run_end_of_round(self, state: PhaseState) -> str:
        state.cumulative_mutations = self._canonicalize_mutations(state.cumulative_mutations)
        artifacts = self.plugin.build_end_of_round_artifacts(state)
        diagnostics_path = self.round_manager.diagnostics_path(self.output_dir) if self.round_manager else self.output_dir / "round_final_diagnostics.txt"
        evaluation_path = self.round_manager.evaluation_path(self.output_dir) if self.round_manager else self.output_dir / "round_evaluation.txt"
        full_diagnostics = getattr(self.plugin, "write_full_diagnostics", None)
        full_diagnostics_required = _plugin_requires_full_diagnostics(self.plugin)
        if full_diagnostics_required and not callable(full_diagnostics):
            raise RuntimeError(f"{self.plugin.name} requires full diagnostics but does not implement write_full_diagnostics")
        if not callable(full_diagnostics):
            diagnostics_path.write_text(artifacts.final_diagnostics_text, encoding="utf-8")
        report = build_end_of_round_report(self.plugin.name, state, artifacts)
        evaluation_path.write_text(report, encoding="utf-8")
        final_metrics = self.plugin.compute_final_metrics(state.cumulative_mutations)
        full_diagnostics_payload = None
        diagnostics_mode = "plugin_full_diagnostics" if callable(full_diagnostics) else "shared_end_of_round_artifacts"
        if callable(full_diagnostics):
            full_diagnostics_payload = full_diagnostics(
                state,
                self.output_dir,
                round_num=self.round_num,
                round_name=self.round_name,
            )
        _ensure_round_final_diagnostics(
            self.plugin,
            diagnostics_path,
            mode=diagnostics_mode,
        )
        diagnostics_status = _write_round_diagnostics_status(
            self.plugin,
            self.output_dir,
            diagnostics_path,
            evaluation_path,
            mode=diagnostics_mode,
            payload=full_diagnostics_payload,
        )
        if self.round_manager and self.round_num is not None:
            artifact_metadata = _plugin_artifact_metadata(self.plugin, state, final_metrics)
            artifact_metadata["final_diagnostics"] = diagnostics_status
            previous_guard = self._previous_promotion_guard(final_metrics)
            if previous_guard:
                artifact_metadata["previous_promotion_guard"] = previous_guard
                if not previous_guard.get("passed", False):
                    raise RuntimeError(
                        f"{self.plugin.name} round {self.round_num} final primary promotion value "
                        f"{previous_guard.get('actual_value')!r} is below the previous optimized baseline "
                        f"{previous_guard.get('expected_value')!r}; refusing to write optimized_config.json"
                    )
            self.round_manager.write_run_summary(
                self.output_dir,
                state.cumulative_mutations,
                final_metrics,
                state.completed_phases,
                round_num=self.round_num,
                artifact_metadata=artifact_metadata,
            )
            self.round_manager.write_optimized_config(self.output_dir, state.cumulative_mutations, artifact_metadata=artifact_metadata)
            self.round_manager.append_to_manifest(
                self.round_num,
                state.cumulative_mutations,
                final_metrics,
                artifact_metadata=artifact_metadata,
            )
        self.phase_logger.log_activity(max(state.completed_phases, default=0), "end_of_round", {"completed_phases": state.completed_phases})
        return report

    def _prepare_state_for_phase(self, state: PhaseState, phase: int) -> PhaseState:
        stale = _phases_at_or_after(state, phase)
        if not stale:
            state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
            return state
        self.phase_logger.backup_state(self.state_path, f"pre_phase_{phase}_rerun")
        self.phase_logger.clear_generated_outputs(phase)
        for container in (state.phase_results, state.phase_gate_results, state.retry_count, state.scoring_retries, state.diagnostic_retries, state.phase_timestamps):
            for stale_phase in stale:
                container.pop(stale_phase, None)
        state.completed_phases = [item for item in state.completed_phases if item < phase]
        state.current_phase = max(state.completed_phases, default=0)
        state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
        self.phase_logger.prune_progress(set(state.completed_phases), current_phase=state.current_phase)
        save_phase_state(state, self.state_path)
        return state

    def _base_mutations_for_phase(self, state: PhaseState, phase: int) -> dict[str, Any]:
        mutations = self._initial_baseline_mutations()
        for phase_num in sorted(state.phase_results):
            if phase_num >= phase:
                break
            mutations.update(state.phase_results[phase_num].get("new_mutations", {}))
        return self._canonicalize_mutations(mutations)

    def _initial_baseline_mutations(self) -> dict[str, Any]:
        explicit = getattr(self.plugin, "initial_mutations_override", None)
        if isinstance(explicit, dict):
            return self._canonicalize_mutations(explicit)
        if self.round_manager and self.round_num and self.round_num > 1:
            plugin_context = getattr(self.plugin, "execution_context", {}) or {}
            previous = self.round_manager.get_previous_mutations(
                self.round_num,
                expected_execution_contract=build_execution_contract(self.plugin) if plugin_context else None,
                allow_incompatible_baseline=bool(getattr(self.plugin, "config", {}).get("allow_incompatible_baseline", False))
                if isinstance(getattr(self.plugin, "config", {}), dict)
                else False,
            )
            if previous:
                return self._canonicalize_mutations(previous)
        return self._canonicalize_mutations(dict(getattr(self.plugin, "initial_mutations", None) or {}))

    def _initial_baseline_source(self) -> str | None:
        config = getattr(self.plugin, "config", {}) or {}
        if isinstance(config, dict) and config.get("initial_mutations_path"):
            path = Path(str(config["initial_mutations_path"]))
            return str(path if path.is_absolute() else Path.cwd() / path)
        if self.round_manager and self.round_num and self.round_num > 1:
            path = self.round_manager.previous_optimized_config_path(self.round_num)
            return str(path) if path else None
        return None

    def _previous_optimized_config(self) -> dict[str, Any]:
        if not self.round_manager or not self.round_num or self.round_num <= 1:
            return {}
        previous = self.round_manager.previous_active_round(self.round_num)
        return self.round_manager.load_optimized_config(previous) if previous >= 1 else {}

    def _validate_initial_baseline(self, mutations: dict[str, Any]) -> dict[str, Any]:
        config = getattr(self.plugin, "config", {}) or {}
        if not isinstance(config, dict) or not bool(config.get("baseline_integrity_required", False)):
            return {}
        previous = self._previous_optimized_config()
        if not previous:
            raise RuntimeError(f"{self.plugin.name} baseline integrity requires a previous optimized_config.json, but none was found")
        previous_contract = previous.get("metric_contract") if isinstance(previous.get("metric_contract"), dict) else {}
        metric_name = str(previous.get("primary_promotion_metric") or previous_contract.get("primary_promotion_metric") or "official_mtm_net_return_pct")
        expected = previous.get("primary_promotion_value", previous_contract.get("primary_promotion_value", previous.get(metric_name)))
        if expected is None:
            raise RuntimeError(f"{self.plugin.name} previous optimized config is missing primary promotion value for {metric_name}")
        metrics = self.plugin.compute_final_metrics(self._canonicalize_mutations(mutations))
        actual = metrics.get(metric_name, metrics.get("primary_promotion_value"))
        tolerance = float(config.get("baseline_integrity_tolerance_abs", 0.001) or 0.0)
        expected_f = float(expected)
        actual_f = float(actual)
        delta = actual_f - expected_f
        passed = abs(delta) <= tolerance
        report = {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metric": metric_name,
            "expected_value": expected_f,
            "actual_value": actual_f,
            "delta": delta,
            "tolerance_abs": tolerance,
            "baseline_source": self._initial_baseline_source(),
            "previous_strategy_core_version": previous.get("strategy_core_version")
            or (previous.get("execution_contract") if isinstance(previous.get("execution_contract"), dict) else {}).get("strategy_core_version"),
            "current_strategy_core_version": metrics.get("strategy_core_version"),
        }
        if not passed:
            raise RuntimeError(
                f"{self.plugin.name} initial baseline drifted before optimization: {metric_name} "
                f"expected {expected_f:.6f}, got {actual_f:.6f} (tolerance {tolerance:.6f}). "
                "Structural changes must preserve the previous optimized baseline or be run as explicit experiments."
            )
        return report

    def _previous_promotion_guard(self, metrics: dict[str, Any]) -> dict[str, Any]:
        config = getattr(self.plugin, "config", {}) or {}
        if not isinstance(config, dict) or not bool(config.get("previous_promotion_guard_required", False)):
            return {}
        previous = self._previous_optimized_config()
        if not previous:
            return {"status": "missing_previous_optimized_config", "passed": False}
        previous_contract = previous.get("metric_contract") if isinstance(previous.get("metric_contract"), dict) else {}
        metric_name = str(previous.get("primary_promotion_metric") or previous_contract.get("primary_promotion_metric") or metrics.get("primary_promotion_metric") or "official_mtm_net_return_pct")
        expected = previous.get("primary_promotion_value", previous_contract.get("primary_promotion_value", previous.get(metric_name)))
        actual = metrics.get(metric_name, metrics.get("primary_promotion_value"))
        if expected is None or actual is None:
            return {"status": "missing_primary_value", "passed": False, "metric": metric_name}
        tolerance = float(config.get("previous_promotion_tolerance_abs", config.get("baseline_integrity_tolerance_abs", 0.001)) or 0.0)
        expected_f = float(expected)
        actual_f = float(actual)
        delta = actual_f - expected_f
        passed = delta >= -tolerance
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metric": metric_name,
            "expected_value": expected_f,
            "actual_value": actual_f,
            "delta": delta,
            "tolerance_abs": tolerance,
        }

    def _canonicalize_mutations(self, mutations: dict[str, Any] | None) -> dict[str, Any]:
        hook = getattr(self.plugin, "canonicalize_mutations", None)
        if callable(hook):
            return dict(hook(dict(mutations or {})))
        return dict(mutations or {})

    def _progress(self, state: PhaseState, phase: int, status: str, focus: str, candidate_count: int, extra: dict[str, Any] | None = None) -> None:
        summary = {
            "status": status,
            "updated_at": _utc_now_iso(),
            "completed_phases": list(state.completed_phases),
            "current_phase": phase,
            "phase": phase,
            "focus": focus,
            "candidate_count": candidate_count,
            "total_mutations": len(state.cumulative_mutations),
            "scoring_retries": state.scoring_retries.get(phase, 0),
            "diagnostic_retries": state.diagnostic_retries.get(phase, 0),
        }
        if extra:
            summary.update(extra)
        self.phase_logger.update_progress(phase, summary)


def _to_dict(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _gate_to_dict(gate: GateResult) -> dict:
    return {
        "passed": gate.passed,
        "criteria": [_to_dict(item) for item in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    }


def _apply_phase_metric_contract(
    metrics: dict[str, Any] | None,
    spec: PhaseSpec,
    plugin: StrategyPlugin,
) -> dict[str, Any]:
    """Attach the declared phase metric contract before scoring or promotion gates."""
    result: dict[str, Any] = dict(metrics or {})
    contract: dict[str, Any] = dict(result.get("metric_contract") or {})
    if spec.phase_metric_basis:
        result.setdefault("phase_metric_basis", spec.phase_metric_basis)
    if spec.primary_promotion_metric:
        result["primary_promotion_metric"] = spec.primary_promotion_metric
    primary_metric = str(
        result.get("primary_promotion_metric")
        or contract.get("primary_promotion_metric")
        or ""
    )
    if primary_metric and "primary_promotion_value" not in result:
        result["primary_promotion_value"] = result.get(primary_metric)
    if spec.official_metric_keys:
        result["official_metric_keys"] = list(spec.official_metric_keys)
        contract["official_metrics"] = list(spec.official_metric_keys)
    if spec.proxy_metric_keys:
        result["proxy_metric_keys"] = list(spec.proxy_metric_keys)
        contract["proxy_metrics"] = list(spec.proxy_metric_keys)
    if spec.primary_promotion_metric:
        contract["primary_promotion_metric"] = spec.primary_promotion_metric
        contract["primary_promotion_value"] = result.get("primary_promotion_value")
    if spec.phase_metric_basis:
        contract["phase_metric_basis"] = spec.phase_metric_basis
    if spec.promotion_requires_audit_pass:
        result["promotion_requires_audit_pass"] = True
    contract["promotion_requires_audit_pass"] = bool(
        result.get("promotion_requires_audit_pass", contract.get("promotion_requires_audit_pass", False))
    )
    if result.get("primary_promotion_basis") and "primary_promotion_basis" not in contract:
        contract["primary_promotion_basis"] = result["primary_promotion_basis"]
    if "execution_contract" not in result and not isinstance(contract.get("execution_contract"), dict):
        execution_contract = build_execution_contract(plugin, result)
        result["execution_contract"] = execution_contract
        contract["execution_contract"] = execution_contract
    elif isinstance(result.get("execution_contract"), dict):
        contract.setdefault("execution_contract", result["execution_contract"])
    result["metric_contract"] = contract
    return result


def _audit_contract_criteria(metrics: dict[str, Any]) -> list[GateCriterion]:
    return _metric_contract_criteria(metrics)


def _plugin_phase_acceptance_criteria(
    plugin: StrategyPlugin,
    phase: int,
    base_mutations: dict[str, Any],
    final_metrics: dict[str, Any],
    greedy_result: GreedyResult,
) -> list[GateCriterion]:
    hook = getattr(plugin, "phase_acceptance_criteria", None)
    if not callable(hook):
        return []
    base_metrics = plugin.compute_final_metrics(base_mutations)
    criteria = hook(
        phase=phase,
        base_mutations=base_mutations,
        base_metrics=base_metrics,
        final_metrics=final_metrics,
        greedy_result=greedy_result,
    )
    return list(criteria or [])


def _metric_contract_criteria(metrics: dict[str, Any]) -> list[GateCriterion]:
    metrics = metrics or {}
    contract = metrics.get("metric_contract") if isinstance(metrics.get("metric_contract"), dict) else {}
    primary_metric = str(metrics.get("primary_promotion_metric", contract.get("primary_promotion_metric", "")) or "")
    requires = bool(metrics.get("promotion_requires_audit_pass", contract.get("promotion_requires_audit_pass", False)))
    contract_active = bool(contract or primary_metric or requires or metrics.get("official_metric_basis"))
    if not contract_active:
        return []
    primary_value = metrics.get(primary_metric, contract.get("primary_promotion_value")) if primary_metric else contract.get("primary_promotion_value")
    primary_is_finite = _is_finite(primary_value)
    official_basis = str(metrics.get("official_metric_basis", contract.get("primary_promotion_basis", "")) or "")
    criteria = [
        GateCriterion("hard_primary_promotion_metric", 1.0, 1.0 if primary_is_finite else 0.0, primary_is_finite),
        GateCriterion("hard_official_metric_basis", 1.0, 1.0 if official_basis else 0.0, bool(official_basis)),
    ]
    official_metric_keys = list(
        metrics.get("official_metric_keys")
        or contract.get("official_metrics")
        or ([primary_metric] if primary_metric else [])
    )
    official_metric_present = any(
        bool(key) and key in metrics and _is_finite(metrics.get(key))
        for key in official_metric_keys
    )
    criteria.append(
        GateCriterion(
            "hard_official_metric_present",
            1.0,
            1.0 if official_metric_present else 0.0,
            official_metric_present,
        )
    )
    required_hygiene = set(
        contract.get(
            "required_hygiene_metrics",
            (
                "same_bar_fill_count",
                "forced_replay_close_count",
                "rejected_order_count",
                "end_open_position_count",
            ),
        )
        or ()
    )
    allow_missing_hygiene = bool(contract.get("allow_missing_hygiene_metrics", metrics.get("allow_missing_hygiene_metrics", False)))
    for metric_name, tolerance_name, default_tolerance in (
        ("same_bar_fill_count", "max_same_bar_fills", 0.0),
        ("forced_replay_close_count", "max_forced_replay_closes", 0.0),
        ("rejected_order_count", "max_rejected_orders", 0.0),
    ):
        present = metric_name in metrics or metric_name in contract
        tolerance = float(metrics.get(tolerance_name, contract.get(tolerance_name, default_tolerance)) or 0.0)
        actual = float(metrics.get(metric_name, contract.get(metric_name, 0.0)) or 0.0)
        passed = (present or metric_name not in required_hygiene or allow_missing_hygiene) and actual <= tolerance
        criteria.append(GateCriterion(f"hard_{metric_name}", tolerance, actual, passed))
    open_tolerance = float(metrics.get("max_end_open_positions", contract.get("max_end_open_positions", 0.0)) or 0.0)
    end_open_present = "end_open_position_count" in metrics or "end_open_position_count" in contract
    end_open = float(metrics.get("end_open_position_count", contract.get("end_open_position_count", 0.0)) or 0.0)
    allow_open_with_mtm = bool(metrics.get("allow_end_open_positions_with_mtm", contract.get("allow_end_open_positions_with_mtm", False)))
    end_open_evidence = end_open_present or "end_open_position_count" not in required_hygiene or allow_missing_hygiene
    end_open_passed = end_open_evidence and (end_open <= open_tolerance or (allow_open_with_mtm and bool(official_basis)))
    criteria.append(GateCriterion("hard_end_open_position_count", open_tolerance, end_open, end_open_passed))
    for field in ("source_fingerprint", "feature_manifest_hash", "candidate_snapshot_hash"):
        value = metrics.get(field) or contract.get(field)
        if not value and isinstance(contract.get("execution_contract"), dict):
            value = contract["execution_contract"].get(field)
        criteria.append(GateCriterion(f"hard_{field}", 1.0, 1.0 if value else 0.0, bool(value)))

    audit_status = str(metrics.get("audit_status", contract.get("audit_status", "")) or "").lower()
    audit_pass = bool(
        metrics.get("audit_pass")
        or contract.get("audit_pass")
        or metrics.get("full_official_audit_passed")
        or metrics.get("fast_full_audit_passed")
        or audit_status in {"pass", "passed"}
    )
    if audit_status in {"fail", "failed", "missing", "not_run"}:
        audit_pass = False
    if requires:
        criteria.append(GateCriterion("hard_audit_pass", 1.0, 1.0 if audit_pass else 0.0, audit_pass))
        execution_contract = contract.get("execution_contract") if isinstance(contract.get("execution_contract"), dict) else {}
        capability_level = str(
            metrics.get("capability_level")
            or contract.get("capability_level")
            or execution_contract.get("capability_level")
            or ""
        ).lower()
        synthetic_levels = {"synthetic", "fixture", "mock", "test"}
        allow_synthetic = bool(metrics.get("allow_synthetic_promotion", contract.get("allow_synthetic_promotion", False)))
        capability_ok = bool(capability_level) and (capability_level not in synthetic_levels or allow_synthetic)
        criteria.append(
            GateCriterion(
                "hard_non_synthetic_capability",
                1.0,
                1.0 if capability_ok else 0.0,
                capability_ok,
            )
        )
    return criteria


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _analysis_to_dict(analysis: PhaseAnalysis) -> dict:
    return {
        "phase": analysis.phase,
        "goal_progress": analysis.goal_progress,
        "strengths": analysis.strengths,
        "weaknesses": analysis.weaknesses,
        "scoring_assessment": analysis.scoring_assessment,
        "diagnostic_gaps": analysis.diagnostic_gaps,
        "suggested_experiments": [_to_dict(item) for item in analysis.suggested_experiments],
        "recommendation": analysis.recommendation,
        "recommendation_reason": analysis.recommendation_reason,
        "report": analysis.report,
        "scoring_weight_overrides": analysis.scoring_weight_overrides,
        "extra": analysis.extra,
    }


def _plugin_execution_context(plugin: StrategyPlugin) -> dict[str, Any]:
    context: dict[str, Any] = dict(getattr(plugin, "execution_context", {}) or {})
    for attr in ("data_dir", "initial_equity", "start_date", "end_date", "max_workers", "capability_level"):
        if hasattr(plugin, attr):
            value = getattr(plugin, attr)
            context[attr] = str(value) if isinstance(value, Path) else value
    context["execution_contract"] = build_execution_contract(plugin)
    return context


def _plugin_requires_full_diagnostics(plugin: StrategyPlugin) -> bool:
    config = dict(getattr(plugin, "config", {}) or {})
    return bool(getattr(plugin, "requires_full_diagnostics", False) or config.get("require_full_diagnostics", False))


def _ensure_round_final_diagnostics(plugin: StrategyPlugin, diagnostics_path: Path, *, mode: str) -> None:
    path = Path(diagnostics_path)
    if not path.exists():
        raise RuntimeError(
            f"{getattr(plugin, 'name', '')} end-of-round diagnostics did not create {path.name} "
            f"(mode={mode})"
        )
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(
            f"{getattr(plugin, 'name', '')} end-of-round diagnostics wrote an empty {path.name} "
            f"(mode={mode})"
        )


def _write_round_diagnostics_status(
    plugin: StrategyPlugin,
    output_dir: Path,
    diagnostics_path: Path,
    evaluation_path: Path,
    *,
    mode: str,
    payload: Any = None,
) -> dict[str, Any]:
    diagnostics_path = Path(diagnostics_path)
    evaluation_path = Path(evaluation_path)
    report_bytes = diagnostics_path.stat().st_size if diagnostics_path.exists() else 0
    if diagnostics_path.exists():
        report_lines = len(diagnostics_path.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        report_lines = 0
    status = {
        "strategy": getattr(plugin, "name", ""),
        "generated_at_utc": _utc_now_iso(),
        "mode": mode,
        "round_final_diagnostics_path": str(diagnostics_path),
        "round_final_diagnostics_exists": diagnostics_path.exists(),
        "round_final_diagnostics_bytes": report_bytes,
        "round_final_diagnostics_lines": report_lines,
        "round_evaluation_path": str(evaluation_path),
        "round_evaluation_exists": evaluation_path.exists(),
        "plugin_full_diagnostics_callable": callable(getattr(plugin, "write_full_diagnostics", None)),
        "plugin_full_diagnostics_required": _plugin_requires_full_diagnostics(plugin),
        "payload": _diagnostics_payload_summary(payload),
    }
    _atomic_write_json(status, Path(output_dir) / "round_final_diagnostics_status.json")
    return status


def _diagnostics_payload_summary(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, list):
        return payload if len(json.dumps(payload, default=str)) <= 4000 else {"type": "list", "count": len(payload)}
    if isinstance(payload, dict):
        encoded = json.dumps(payload, sort_keys=True, default=str)
        if len(encoded) <= 4000:
            return payload
        summary: dict[str, Any] = {
            "type": "dict",
            "key_count": len(payload),
            "keys": sorted(str(key) for key in payload)[:50],
        }
        for key in (
            "strategy",
            "round",
            "round_name",
            "generated_at_utc",
            "promotion_status",
            "selected_candidate",
            "source_fingerprint",
            "report_path",
        ):
            if key in payload:
                summary[key] = payload[key]
        return summary
    return str(payload)


def _plugin_artifact_metadata(plugin: StrategyPlugin, state: PhaseState, final_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(getattr(plugin, "config", {}) or {})
    context = _plugin_execution_context(plugin)
    metrics = dict(final_metrics or {})
    execution_contract = build_execution_contract(plugin, metrics)
    promotion_status = str(config.get("promotion_status") or "research_only")
    capability_level = str(getattr(plugin, "capability_level", config.get("capability_level", "synthetic")))
    metadata = {
        "promotion_status": promotion_status,
        "artifact_promotion_policy": config.get("artifact_promotion_policy", "research_only_until_feature_complete"),
        "capability_level": capability_level,
        "source_data_fingerprint": getattr(plugin, "source_fingerprint", ""),
        "score_spec_hash": _stable_hash({
            "phase_results": state.phase_results,
            "phase_gate_results": state.phase_gate_results,
        }),
        "config_hash": _stable_hash(config),
        "strategy_code_hash": _strategy_code_hash(plugin),
        "live_parity_fill_timing": config.get("live_parity_fill_timing", "next_bar_after_completed_signal"),
        "risk_basis": config.get("risk_basis", context.get("risk_basis", "mark_to_market")),
        "metric_contract": metrics.get("metric_contract", {}),
        "execution_contract": execution_contract,
        "primary_promotion_metric": metrics.get("primary_promotion_metric"),
        "primary_promotion_value": metrics.get("primary_promotion_value"),
        "primary_promotion_basis": metrics.get("primary_promotion_basis"),
        "official_replay_pass": metrics.get("official_replay_pass"),
        "audit_pass": metrics.get("audit_pass"),
        "audit_status": metrics.get("audit_status"),
    }
    passthrough = (
        "shared_decision_core",
        "strategy_core_version",
        "source_fingerprint",
        "feature_manifest_hash",
        "candidate_snapshot_hash",
        "live_parity_fill_timing",
        "auction_mode",
        "artifact_promotion_policy",
        "account_scope",
        "diagnostics_version",
        "phase_analyzer_version",
        "raw_metric_cache_key",
        "phase_score_cache_key",
        "resume_checkpoint_id",
        "official_promotion_gate",
        "replay_mode",
        "risk_basis",
        "configured_universe_size",
        "holdout_weeks",
        "holdout_start",
        "train_start",
        "train_end",
    )
    for key in passthrough:
        if context.get(key) not in (None, ""):
            metadata[key] = context[key]
    if context.get("source_fingerprint") and not metadata.get("source_data_fingerprint"):
        metadata["source_data_fingerprint"] = context["source_fingerprint"]
    return metadata


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strategy_code_hash(plugin: StrategyPlugin) -> str:
    try:
        path = Path(inspect.getfile(plugin.__class__))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _phases_at_or_after(state: PhaseState, phase: int) -> list[int]:
    phases = set()
    for container in (state.phase_results, state.phase_gate_results, state.retry_count, state.scoring_retries, state.diagnostic_retries, state.phase_timestamps):
        phases.update(key for key in container if key >= phase)
    phases.update(item for item in state.completed_phases if item >= phase)
    return sorted(phases)


def _round_is_complete(state: PhaseState, num_phases: int) -> bool:
    expected = set(range(1, int(num_phases) + 1))
    return bool(expected) and expected.issubset(set(state.completed_phases))


def _dedupe_experiments(experiments: list) -> list:
    seen: set[str] = set()
    result = []
    for item in experiments:
        if item.name in seen:
            continue
        seen.add(item.name)
        result.append(item)
    return result
