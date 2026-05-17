from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .evaluation import build_end_of_round_report
from .greedy_optimizer import run_greedy
from .phase_analyzer import analyze_phase
from .phase_gates import evaluate_gate
from .phase_logging import PhaseLogger
from .phase_state import PhaseState, _atomic_write_json, _utc_now_iso, load_phase_state, save_phase_state
from .plugin import StrategyPlugin
from .round_manager import RoundManager
from .types import GateResult, PhaseAnalysis


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
        if self.round_name and not state.round_name:
            state.round_name = self.round_name
        return state

    def run_all_phases(self, start_phase: int | None = None) -> PhaseState:
        state = self.load_state()
        save_phase_state(state, self.state_path)
        if self.round_manager and self.round_num is not None:
            self.round_manager.write_run_spec(
                self.output_dir,
                self.round_num,
                self.plugin.name,
                description=self.round_name or f"Round {self.round_num}",
                baseline_mutations=dict(state.cumulative_mutations),
                execution_context=_plugin_execution_context(self.plugin),
            )
        start = start_phase or (max(state.completed_phases) + 1 if state.completed_phases else 1)
        try:
            for phase in range(start, self.plugin.num_phases + 1):
                state = self.run_phase(phase, state)
            if not (self.output_dir / "round_evaluation.txt").exists():
                self.run_end_of_round(state)
        finally:
            close_pool = getattr(self.plugin, "close_pool", None)
            if callable(close_pool):
                close_pool()
        return state

    def run_phase(self, phase: int, state: PhaseState | None = None) -> PhaseState:
        state = self._prepare_state_for_phase(state or self.load_state(), phase)
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
                metrics = self.plugin.compute_final_metrics(greedy_result.final_mutations)
                greedy_result.final_metrics = dict(metrics or {})
                self.phase_logger.save_phase_output(phase, "greedy", _to_dict(greedy_result))
                gate_result = evaluate_gate(spec.gate_criteria_fn(metrics), greedy_result)
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
            adopted_mutations = dict(greedy_result.final_mutations if applied else base_mutations)
            adopted_metrics = metrics if applied else self.plugin.compute_final_metrics(base_mutations)
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
        artifacts = self.plugin.build_end_of_round_artifacts(state)
        diagnostics_path = self.round_manager.diagnostics_path(self.output_dir) if self.round_manager else self.output_dir / "round_final_diagnostics.txt"
        evaluation_path = self.round_manager.evaluation_path(self.output_dir) if self.round_manager else self.output_dir / "round_evaluation.txt"
        diagnostics_path.write_text(artifacts.final_diagnostics_text, encoding="utf-8")
        report = build_end_of_round_report(self.plugin.name, state, artifacts)
        evaluation_path.write_text(report, encoding="utf-8")
        final_metrics = self.plugin.compute_final_metrics(state.cumulative_mutations)
        if self.round_manager and self.round_num is not None:
            artifact_metadata = _plugin_artifact_metadata(self.plugin, state)
            self.round_manager.write_run_summary(
                self.output_dir,
                state.cumulative_mutations,
                final_metrics,
                state.completed_phases,
                round_num=self.round_num,
                artifact_metadata=artifact_metadata,
            )
            self.round_manager.write_optimized_config(self.output_dir, state.cumulative_mutations, artifact_metadata=artifact_metadata)
            self.round_manager.append_to_manifest(self.round_num, state.cumulative_mutations, final_metrics)
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
        return mutations

    def _initial_baseline_mutations(self) -> dict[str, Any]:
        if self.round_manager and self.round_num and self.round_num > 1:
            previous = self.round_manager.get_previous_mutations(self.round_num)
            if previous:
                return previous
        return dict(getattr(self.plugin, "initial_mutations", None) or {})

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
    context: dict[str, Any] = {}
    for attr in ("data_dir", "initial_equity", "start_date", "end_date", "max_workers", "capability_level"):
        if hasattr(plugin, attr):
            value = getattr(plugin, attr)
            context[attr] = str(value) if isinstance(value, Path) else value
    return context


def _plugin_artifact_metadata(plugin: StrategyPlugin, state: PhaseState) -> dict[str, Any]:
    config = dict(getattr(plugin, "config", {}) or {})
    promotion_status = str(config.get("promotion_status") or "research_only")
    capability_level = str(getattr(plugin, "capability_level", config.get("capability_level", "synthetic")))
    return {
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
        "risk_basis": "mark_to_market",
    }


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


def _dedupe_experiments(experiments: list) -> list:
    seen: set[str] = set()
    result = []
    for item in experiments:
        if item.name in seen:
            continue
        seen.add(item.name)
        result.append(item)
    return result
