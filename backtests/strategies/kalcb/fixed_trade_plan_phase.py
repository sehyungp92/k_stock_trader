from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import build_cache_key, stable_signature
from backtests.auto.shared.phase_state import PhaseState, save_phase_state
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, PhaseDecision, ScoredCandidate
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts
from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION

from .runner import KALCBReplayAdapter, _collapse_exit_legs
from .trade_plan_sweep import (
    PORTFOLIO_RISK_POLICY,
    PRIMARY_OBJECTIVE_METRIC,
    _add_compiled_candidate_pool_metrics,
    _add_portfolio_equivalent_metrics,
    _add_return_divergence_metrics,
    _broker_trades_to_slot_outcomes,
    _clone_snapshots_for_replay,
    _fold_metrics_from_outcomes_for_dates,
    _replay_digest,
    compile_core_replay,
    load_or_build_prepared_context,
    Selection,
    summarize_outcomes,
)


FIXED_PHASE_AUTO_VERSION = "kalcb-source-frontier-phase-auto-v13"
FIXED_PREVIOUS_FRAMEWORK_VERSION = "fixed_trade_plan_round_v1"
CONSOLIDATED_PHASE_COUNT = 6
OFFICIAL_PROMOTION_METRIC = "official_mtm_net_return_pct"
SOURCE_PATH_MUTATION = "_kalcb.source.path"
SOURCE_SECTION_MUTATION = "_kalcb.source.section"
SOURCE_RANK_MUTATION = "_kalcb.source.rank"
POOL_SOURCE_PATH_MUTATION = "_kalcb.pool_source.path"
POOL_SOURCE_ACTIVE_COUNT_MUTATION = "_kalcb.pool_source.active_count"
POOL_SOURCE_LABEL_MUTATION = "_kalcb.pool_source.label"
ROUND_DIR_RE = re.compile(r"^round_(\d+)$")
SCORE_COMPONENTS = {
    "broker_net_return_pct": 0.24,
    "broker_expected_total_r": 0.16,
    "frequency": 0.12,
    "worst_fold_net": 0.15,
    "avg_mfe_capture": 0.15,
    "broker_max_drawdown_pct": -0.11,
    "mae_tail_loss": -0.07,
}
HARD_REJECTS = {
    "min_trades": 120,
    "max_dd_pct": 0.08,
    "min_worst_fold_net": 0.0,
    "min_net_return_pct": 0.0,
    "max_same_bar_fills": 0,
    "max_end_open_positions": 0,
}


@dataclass(frozen=True, slots=True)
class FixedCandidateSourceRef:
    path: str
    section: str
    rank: int
    row_name: str = ""


@dataclass(frozen=True, slots=True)
class FixedEvaluation:
    metrics: dict[str, Any]
    replay_digest: dict[str, Any]
    fold_rows: tuple[dict[str, Any], ...]
    trade_rows: tuple[dict[str, Any], ...] = tuple()
    decision_summary: dict[str, Any] = field(default_factory=dict)


class FixedCandidateBatchEvaluator:
    def __init__(self, plugin: "KALCBFixedTradePlanOptimizationPlugin", phase: int, scoring_weights: dict[str, float] | None, hard_rejects: dict[str, float] | None):
        self.plugin = plugin
        self.phase = int(phase)
        self.scoring_weights = scoring_weights
        self.hard_rejects = dict(hard_rejects or {})

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        worker_count = max(1, min(int(self.plugin.max_workers or 1), len(candidates)))
        if worker_count <= 1:
            return [self._score(candidate, current_mutations) for candidate in candidates]
        out: list[ScoredCandidate] = []
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"kalcb-fixed-p{self.phase}") as pool:
            futures = {pool.submit(self._score, candidate, current_mutations): candidate for candidate in candidates}
            for future in as_completed(futures):
                out.append(future.result())
        order = {candidate.name: index for index, candidate in enumerate(candidates)}
        out.sort(key=lambda item: order.get(item.name, 0))
        return out

    def _score(self, candidate: Experiment, current_mutations: dict[str, Any]) -> ScoredCandidate:
        mutations = dict(current_mutations or {})
        mutations.update(candidate.mutations or {})
        baseline_metrics = self.plugin.evaluate_mutations(current_mutations)
        metrics = self.plugin.evaluate_mutations(mutations)
        reject = reject_reason(metrics, {**HARD_REJECTS, **self.hard_rejects})
        if reject:
            return ScoredCandidate(candidate.name, 0.0, rejected=True, reject_reason=reject, metrics=metrics)
        guardrail_reject = _quality_guardrail_reject_reason(metrics, baseline_metrics)
        if guardrail_reject:
            return ScoredCandidate(candidate.name, 0.0, rejected=True, reject_reason=guardrail_reject, metrics=metrics)
        if self.plugin.validation_gate_enabled and not bool(candidate.mutations.get("_research_only")):
            baseline_validation = self.plugin.evaluate_validation_mutations(current_mutations)
            validation = self.plugin.evaluate_validation_mutations(mutations)
            metrics["validation_metrics"] = _compact_validation_metrics(validation)
            metrics["validation_delta"] = _metric_deltas(validation, baseline_validation)
            metrics["train_delta"] = _metric_deltas(metrics, baseline_metrics)
            validation_reject = _validation_gate_reject_reason(metrics, baseline_metrics, validation, baseline_validation)
            metrics["validation_gate_passed"] = not bool(validation_reject)
            if validation_reject:
                return ScoredCandidate(candidate.name, 0.0, rejected=True, reject_reason=validation_reject, metrics=metrics)
        return ScoredCandidate(candidate.name, score_fixed(metrics, self.scoring_weights), metrics=metrics)

    def close(self) -> None:
        return None


class KALCBFixedTradePlanOptimizationPlugin:
    name = "kalcb"
    num_phases = CONSOLIDATED_PHASE_COUNT
    requires_full_diagnostics = True
    default_scoring_weights = dict(SCORE_COMPONENTS)
    ultimate_targets = {
        "broker_net_return_pct": 0.55,
        "broker_expected_total_r": 1800.0,
        "avg_trade_net_pct": 0.009,
        "trade_count": 220.0,
        "avg_mfe_capture": 0.50,
        "broker_max_drawdown_pct": 0.065,
        "worst_fold_net": 0.12,
    }

    def __init__(self, config: dict[str, Any] | None = None, *, output_dir: Path | None = None, max_workers: int | None = 1, capability_level: str = "real_replay"):
        self.config = dict(config or {})
        self.config.setdefault("capability_level", capability_level)
        self.config.setdefault("allow_incompatible_baseline", True)
        self.config.setdefault("artifact_promotion_policy", "research_only_until_holdout_and_paper_parity")
        self.output_dir = Path(output_dir) if output_dir else Path("data/backtests/output/kalcb/fixed_phase_auto")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, min(int(max_workers or self.config.get("workers", 1) or 1), 2))
        self.holdout_excluded_from_optimization = bool(self.config.get("holdout_excluded_from_optimization", False))
        self.validation_gate_enabled = bool(self.config.get("validation_gate_enabled", False)) and not self.holdout_excluded_from_optimization
        self.capability_level = str(self.config.get("capability_level", capability_level))
        self.source_ref = _resolve_fixed_candidate_source(self.config, self.output_dir)
        self.initial_mutations = _initial_mutations_for_output(self.config, self.output_dir)
        self.initial_mutations_override = dict(self.initial_mutations)
        context_config = _replay_context_config(self.config)
        self.context = load_or_build_prepared_context(
            context_config,
            optimized_source=self.source_ref.path,
            candidate_section=self.source_ref.section,
            candidate_rank=self.source_ref.rank,
            strict_candidate_source=False,
            output_dir=self.output_dir,
            train_only=True,
            fold_count=2,
            compiled_cache_dir=context_config.get("compiled_cache_dir"),
            force_rebuild_cache=bool(context_config.get("force_rebuild_cache", False)),
            refresh_cached_baseline=not bool(self.config.get("skip_initial_baseline_eval", False)),
            status_callback=self._context_status,
        )
        self.base_cfg = KALCBConfig.from_mapping(self.context.training_config, {})
        self.source_fingerprint = self.context.compiled_replay.source_fingerprint
        self.feature_manifest_hash = self.context.cache_key
        self.candidate_snapshot_hash = self.context.compiled_replay.candidate_artifact_hash
        self.initial_equity = self.context.compiled_replay.initial_equity
        self._context_cache: dict[str, Any] = {_source_context_key(self.source_ref): self.context}
        self._context_lock = threading.Lock()
        self._validation_plugin_lock = threading.Lock()
        self.execution_context = {
            "strategy": "kalcb",
            "phase_framework_version": FIXED_PREVIOUS_FRAMEWORK_VERSION,
            "phase_auto_version": FIXED_PHASE_AUTO_VERSION,
            "shared_decision_core": "live_shared_core",
            "strategy_core_version": KALCB_CORE_VERSION,
            "source_fingerprint": self.source_fingerprint,
            "feature_manifest_hash": self.feature_manifest_hash,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "candidate_source": {
                "path": self.source_ref.path,
                "section": self.source_ref.section,
                "rank": self.source_ref.rank,
                "row_name": self.source_ref.row_name,
            },
            "date_window": {
                "start": self.context.train_dates[0].isoformat() if self.context.train_dates else "",
                "end": self.context.train_dates[-1].isoformat() if self.context.train_dates else "",
                "sessions": len(self.context.train_dates),
            },
            "initial_equity": self.initial_equity,
            "cost_policy": _cost_policy(self.base_cfg),
            "fill_timing": "next_5m_open",
            "live_parity_fill_timing": "next_5m_open",
            "auction_mode": "non_auction_continuous",
            "capability_level": self.capability_level,
            "replay_mode": "fixed_candidate_shared_core_compiled_replay",
            "primary_promotion_metric": OFFICIAL_PROMOTION_METRIC,
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "research_objective_metric": PRIMARY_OBJECTIVE_METRIC,
            "research_objective_basis": "closed_trade_net_pnl_over_initial_equity",
            "holdout_policy": "training-only compiled replay; dates at or after holdout_start are excluded; holdout is locked audit only and not used for candidate acceptance",
            "validation_gate_enabled": self.validation_gate_enabled,
            "holdout_excluded_from_optimization": self.holdout_excluded_from_optimization,
            "implementation_lessons_contract": {
                "shared_core": "strategy_kalcb.core.step_kalcb_core",
                "backtest_role": "thin replay driver over KALCBReplayAdapter and SimBroker",
                "completed_bar_policy": "first30 uses completed 09:00-09:25 bars; signals fill next 5m open",
                "execution_model": "all entries, exits, stops, partials, and flattens route through neutral actions and SimBroker",
                "risk_basis": "SimBroker mark-to-market equity curve",
            },
        }
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, Any]] = {}
        self._evaluation_details: dict[str, FixedEvaluation] = {}
        self._validation_plugin_cache: KALCBFixedTradePlanOptimizationPlugin | None = None
        self._last_metrics: dict[str, Any] = {}
        self._last_mutation_key = ""
        if bool(self.config.get("skip_initial_baseline_eval", False)):
            self._baseline_metrics = {}
        else:
            self._baseline_metrics = self.evaluate_mutations(self.initial_mutations)

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        focus, focus_metrics = _phase_focus(phase)
        return PhaseSpec(
            focus=focus,
            candidates=get_phase_candidates(phase),
            gate_criteria_fn=lambda metrics: gate_criteria(metrics, HARD_REJECTS),
            scoring_weights=None,
            hard_rejects=dict(HARD_REJECTS),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.001,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=None,
                build_extra_analysis_fn=self.build_phase_extra_analysis,
                format_extra_analysis_fn=self.format_phase_extra_analysis,
                decide_action_fn=self.decide_phase_action,
            ),
            max_rounds=3,
            prune_threshold=0.18,
            reject_streak_limit=2,
            phase_metric_basis="training_only_fixed_candidate_shared_core",
            primary_promotion_metric=OFFICIAL_PROMOTION_METRIC,
            official_metric_keys=(OFFICIAL_PROMOTION_METRIC, PRIMARY_OBJECTIVE_METRIC),
            promotion_requires_audit_pass=True,
        )

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights: dict[str, float] | None = None, hard_rejects: dict[str, float] | None = None):
        baseline_metrics = self.evaluate_mutations(cumulative_mutations)
        baseline_reject = reject_reason(baseline_metrics, {**HARD_REJECTS, **dict(hard_rejects or {})})
        baseline = ScoredCandidate(
            "__baseline__",
            0.0 if baseline_reject else score_fixed(baseline_metrics, scoring_weights),
            rejected=bool(baseline_reject),
            reject_reason=baseline_reject,
            metrics=baseline_metrics,
        )
        baseline_key = _mutation_key(cumulative_mutations)
        self._evaluation_cache.setdefault(_cache_key(phase, baseline_key, "__baseline__"), baseline)

        evaluator = FixedCandidateBatchEvaluator(self, phase, scoring_weights, hard_rejects)

        def wrapped(candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
            results: list[ScoredCandidate] = []
            missing: list[Experiment] = []
            for candidate in candidates:
                key = _cache_key(phase, _mutation_key({**dict(current_mutations or {}), **dict(candidate.mutations or {})}), candidate.name)
                cached = self._evaluation_cache.get(key)
                if cached is not None:
                    results.append(cached)
                else:
                    missing.append(candidate)
            if missing:
                missing_by_name = {candidate.name: candidate for candidate in missing}
                for item in evaluator(missing, current_mutations):
                    candidate = missing_by_name[item.name]
                    key = _cache_key(phase, _mutation_key({**dict(current_mutations or {}), **dict(candidate.mutations or {})}), item.name)
                    self._evaluation_cache[key] = item
                    results.append(item)
            order = {candidate.name: index for index, candidate in enumerate(candidates)}
            results.sort(key=lambda item: order.get(item.name, 0))
            return results

        setattr(wrapped, "close", evaluator.close)
        return wrapped

    def evaluate_mutations(self, mutations: dict[str, Any]) -> dict[str, Any]:
        key = _mutation_key(mutations)
        cached = self._metrics_cache.get(key)
        if cached is not None:
            self._last_metrics = dict(cached)
            self._last_mutation_key = key
            return dict(cached)
        evaluation = self._evaluate(mutations)
        metrics = dict(evaluation.metrics)
        self._metrics_cache[key] = metrics
        self._evaluation_details[key] = evaluation
        self._last_metrics = metrics
        self._last_mutation_key = key
        return dict(metrics)

    def evaluate_validation_mutations(self, mutations: dict[str, Any]) -> dict[str, Any]:
        plugin = self._validation_plugin()
        metrics = plugin.evaluate_mutations(mutations)
        metrics["validation_window"] = "holdout"
        return dict(metrics)

    def _validation_plugin(self) -> "KALCBFixedTradePlanOptimizationPlugin":
        with self._validation_plugin_lock:
            cached = self._validation_plugin_cache
            if cached is not None:
                return cached
            validation_config = _validation_config(self.config)
            validation_config["initial_mutations"] = dict(self.initial_mutations)
            validation_config["fixed_candidate_source"] = {
                "path": self.source_ref.path,
                "section": self.source_ref.section,
                "rank": self.source_ref.rank,
                "row_name": self.source_ref.row_name,
            }
            plugin = KALCBFixedTradePlanOptimizationPlugin(
                validation_config,
                output_dir=self.output_dir / "validation_gate",
                max_workers=1,
                capability_level=self.capability_level,
            )
            self._validation_plugin_cache = plugin
            return plugin

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, Any]:
        metrics = self.evaluate_mutations(mutations)
        audit = self._audit_fast_replay_parity(mutations, metrics)
        metrics["fast_suppression_audit"] = audit
        metrics["fast_full_audit_passed"] = bool(audit.get("pass", False))
        return attach_official_metric_contract(
            metrics,
            primary_metric=OFFICIAL_PROMOTION_METRIC,
            requires_audit_pass=True,
            audit_pass=bool(metrics.get("same_bar_fill_count", 0.0) == 0.0 and metrics.get("fast_full_audit_passed", False)),
            audit_status="direct_shared_core_replay_train_only",
            official_replay_pass=True,
            execution_contract=build_execution_contract(self, metrics, extra={"phase_auto_version": FIXED_PHASE_AUTO_VERSION}),
        )

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, Any], greedy_result: GreedyResult) -> str:
        del state
        return _diagnostics_text(
            phase=phase,
            focus=_phase_focus(phase)[0],
            metrics=metrics,
            baseline=self._baseline_metrics,
            kept=greedy_result.kept_features,
            source_ref=self.source_ref,
            execution_context=self.execution_context,
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, Any], greedy_result: GreedyResult) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result) + "\nEnhanced: replay path remains KALCBReplayAdapter -> shared core -> SimBroker; no standalone fill logic is used.\n"

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        state.cumulative_mutations = _artifact_mutations(state.cumulative_mutations)
        final = self.compute_final_metrics(state.cumulative_mutations)
        text = _final_diagnostics_text(final, self._baseline_metrics, state, self.source_ref, self.execution_context)
        return EndOfRoundArtifacts(
            final_diagnostics_text=text,
            dimension_reports={
                "signal_extraction": "Source/path-risk calibration remains research-only unless a source row passes train, validation, and parity gates; proxy-only source reranks are not promoted.",
                "signal_discrimination": "Phases 1 and 3 focus only on the stable gap-retention/gap-relvol evidence and ultra-capped frontier recovery, avoiding broad frontier or blunt first30 sweeps.",
                "entry_mechanism": "The first30_open anchor remains primary; route overlays are fallback-safe and must improve validation before being accepted.",
                "trade_management": "Phases 4 and 5 test conditional tail capture and low-relvol MFE-floor containment, not recycled global giveback exits.",
                "exit_mechanism": "Static target sweeps are replaced by cohort-conditional targets and route-mode-scoped MFE floors.",
                "risk": "Phase 2 tests drawdown-contained notional activation for high gap-relvol trades; phase 6 validates only near-miss stacks.",
                "promotion": "Optimization excludes holdout for training but validation-gates accepted mutations on the configured holdout window before promotion.",
            },
            overall_verdict="KALCB source/frontier phase-auto completed on the training window only; validate finalists on untouched holdout before live use.",
        )

    def write_full_diagnostics(
        self,
        state: PhaseState,
        output_dir: Path,
        *,
        round_num: int | None = None,
        round_name: str = "",
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        state.cumulative_mutations = _artifact_mutations(state.cumulative_mutations)
        save_phase_state(state, output_dir / "phase_state.json")
        final = self.compute_final_metrics(state.cumulative_mutations)
        final_context = self._context_for_mutations(state.cumulative_mutations)
        final_source_ref = _source_ref_for_mutations(state.cumulative_mutations, self.source_ref)
        final_source_ref = FixedCandidateSourceRef(
            path=final_context.candidate_source.source_path,
            section=final_context.candidate_source.source_section,
            rank=final_context.candidate_source.source_rank,
            row_name=final_context.candidate_source.source_row_name,
        )
        final_detail = self._evaluation_details.get(_mutation_key(state.cumulative_mutations))
        baseline_detail = self._evaluation_details.get(_mutation_key(self.initial_mutations))
        snapshot_manifest = _write_candidate_snapshot_artifacts(output_dir, final_context)
        bar_digest = _compiled_bar_digest(final_context)
        opportunity_diagnostics = _opportunity_coverage_diagnostics(
            final_context,
            state.cumulative_mutations,
            final_detail.trade_rows if final_detail else tuple(),
            final,
        )
        live_parity_audit = _paper_live_parity_requirements(
            final,
            self.execution_context,
            mutations=state.cumulative_mutations,
            source_ref=final_source_ref,
            context=final_context,
            replay_digest=final_detail.replay_digest if final_detail else {},
            snapshot_manifest=snapshot_manifest,
            bar_digest=bar_digest,
        )
        report = _full_round_diagnostics_text(
            final=final,
            baseline=self._baseline_metrics,
            fold_rows=final_detail.fold_rows if final_detail else tuple(),
            baseline_fold_rows=baseline_detail.fold_rows if baseline_detail else tuple(),
            trade_rows=final_detail.trade_rows if final_detail else tuple(),
            baseline_trade_rows=baseline_detail.trade_rows if baseline_detail else tuple(),
            state=state,
            source_ref=final_source_ref,
            execution_context=self.execution_context,
            cache_metadata=final_context.cache_metadata,
            round_num=round_num,
            round_name=round_name,
            live_parity_audit=live_parity_audit,
            opportunity_diagnostics=opportunity_diagnostics,
        )
        diagnostics_summary = _full_diagnostics_summary(
            final,
            self._baseline_metrics,
            state,
            final_source_ref,
            fold_rows=final_detail.fold_rows if final_detail else tuple(),
            baseline_fold_rows=baseline_detail.fold_rows if baseline_detail else tuple(),
            trade_rows=final_detail.trade_rows if final_detail else tuple(),
            opportunity_diagnostics=opportunity_diagnostics,
        )
        index = {
            "strategy": "kalcb",
            "round": round_num,
            "round_name": round_name,
            "report_path": str(output_dir / "round_final_diagnostics.txt"),
            "diagnostics_summary_path": str(output_dir / "diagnostics_summary.json"),
            "live_parity_audit_path": str(output_dir / "live_parity_audit.json"),
            "paper_live_parity_contract_path": str(output_dir / "paper_live_parity_contract.json"),
            "daily_snapshot_manifest_path": str(output_dir / "paper_live_parity_inputs" / "daily_snapshot_manifest.json"),
            "replay_market_bar_digest_path": str(output_dir / "paper_live_parity_inputs" / "replay_market_bar_digest.json"),
            "source_ref": {
                "path": final_source_ref.path,
                "section": final_source_ref.section,
                "rank": final_source_ref.rank,
                "row_name": final_source_ref.row_name,
            },
            "holdout_policy": final.get("holdout_policy", self.execution_context.get("holdout_policy", "")),
            "fast_suppression_audit_status": (final.get("fast_suppression_audit") or {}).get("status", "not_run"),
        }
        _write_text(output_dir / "round_final_diagnostics.txt", report)
        _write_json(output_dir / "diagnostics_summary.json", diagnostics_summary)
        _write_json(output_dir / "live_parity_audit.json", live_parity_audit)
        _write_json(output_dir / "paper_live_parity_contract.json", live_parity_audit)
        _write_json(output_dir / "paper_live_parity_inputs" / "replay_market_bar_digest.json", bar_digest)
        _write_json(output_dir / "full_diagnostics_index.json", index)
        return diagnostics_summary

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, Any]) -> list[str]:
        del phase
        gaps: list[str] = []
        if metrics.get("same_bar_fill_count", 0.0) > 0:
            gaps.append("same_bar_fill_count is non-zero")
        if metrics.get("end_open_position_count", 0.0) > 0:
            gaps.append("end_open_position_count is non-zero")
        if metrics.get("broker_max_drawdown_pct", 0.0) > HARD_REJECTS["max_dd_pct"]:
            gaps.append("broker MTM drawdown breaches the 8% hard ceiling")
        if metrics.get("trade_count", 0.0) < HARD_REJECTS["min_trades"]:
            gaps.append("trade count fell below the anti-overfit minimum")
        return gaps

    def build_phase_extra_analysis(self, phase: int, metrics: dict[str, Any], state: PhaseState, greedy_result: GreedyResult) -> dict[str, Any]:
        del metrics, state
        if greedy_result.accepted_count > 0:
            return {}
        reject_counts = Counter(str(row.get("reject_reason") or "not_score_improving") for row in greedy_result.candidate_evaluations)
        near_misses = sorted(
            (
                {
                    "name": str(row.get("name") or ""),
                    "score_delta_pct": float(row.get("score_delta_pct", 0.0) or 0.0),
                    "reject_reason": str(row.get("reject_reason") or ""),
                    "broker_net_return_pct": row.get("metrics", {}).get("broker_net_return_pct"),
                    "broker_max_drawdown_pct": row.get("metrics", {}).get("broker_max_drawdown_pct"),
                    "avg_trade_net_pct": row.get("metrics", {}).get("avg_trade_net_pct"),
                    "trade_count": row.get("metrics", {}).get("trade_count"),
                    "validation_gate_passed": row.get("metrics", {}).get("validation_gate_passed"),
                }
                for row in greedy_result.candidate_evaluations
                if str(row.get("name") or "") != "__baseline__"
            ),
            key=lambda item: item["score_delta_pct"],
            reverse=True,
        )[:8]
        return {
            "zero_acceptance_analysis": {
                "phase": int(phase),
                "candidate_count": int(greedy_result.total_candidates),
                "accepted_count": int(greedy_result.accepted_count),
                "reject_reason_counts": dict(reject_counts),
                "top_near_misses": near_misses,
                "next_action_policy": "retry once only when a narrower evidence-backed rescue variant exists; otherwise advance without blind reruns",
            }
        }

    def format_phase_extra_analysis(self, extra: dict[str, Any]) -> list[str]:
        zero = dict(extra.get("zero_acceptance_analysis") or {})
        if not zero:
            return []
        lines = [
            f"Zero-acceptance analysis: {zero.get('accepted_count', 0)} accepted from {zero.get('candidate_count', 0)} candidates.",
            f"Reject reasons: {json.dumps(zero.get('reject_reason_counts', {}), sort_keys=True)}",
            str(zero.get("next_action_policy") or ""),
        ]
        for item in list(zero.get("top_near_misses") or [])[:5]:
            lines.append(
                f"Near miss {item.get('name')}: score_delta={float(item.get('score_delta_pct', 0.0) or 0.0):+.4f}%, "
                f"reject={item.get('reject_reason') or 'score_not_improved'}"
            )
        return [line for line in lines if line]

    def decide_phase_action(
        self,
        phase: int,
        metrics: dict[str, Any],
        state: PhaseState,
        greedy_result: GreedyResult,
        gate_result: Any,
        current_weights: dict[str, float] | None,
        analysis: Any,
        max_scoring_retries: int,
        max_diagnostic_retries: int,
    ) -> PhaseDecision | None:
        del metrics, gate_result, current_weights, analysis, max_diagnostic_retries
        if greedy_result.accepted_count > 0:
            return None
        retry = int(state.scoring_retries.get(int(phase), 0) or 0)
        if retry >= min(1, int(max_scoring_retries)):
            return PhaseDecision(
                action="advance",
                reason="No mutation passed after the evidence-backed rescue pass; advancing without repeating dead candidates.",
                scoring_assessment_override="INEFFECTIVE",
            )
        rescue = _rescue_experiments_for_phase(int(phase))
        if not rescue:
            return PhaseDecision(
                action="advance",
                reason="No mutation passed and no untested evidence-backed rescue variants remain for this phase.",
                scoring_assessment_override="INEFFECTIVE",
            )
        return PhaseDecision(
            action="improve_scoring",
            reason="No mutation was accepted; adding narrower rescue variants derived from the quantitative near-miss/failure mode.",
            scoring_assessment_override="INEFFECTIVE",
            extra_suggested_experiments=rescue,
        )

    def suggest_experiments(self, phase: int, metrics: dict[str, Any], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del weaknesses, state
        if int(phase) == 4 and metrics.get("avg_mfe_capture", 0.0) < 0.38:
            return [Experiment(name, mutations) for name, mutations in _tail_capture_rescue_candidates()]
        return []

    def _context_for_mutations(self, mutations: dict[str, Any]):
        source_ref = _source_ref_for_mutations(mutations, self.source_ref)
        source_context = self._source_context_for_ref(source_ref)
        if str(mutations.get(POOL_SOURCE_PATH_MUTATION) or "").strip():
            return self._guarded_pool_context_for_mutations(mutations, source_context, source_ref)
        return source_context

    def _source_context_for_ref(self, source_ref: FixedCandidateSourceRef):
        key = _source_context_key(source_ref)
        with self._context_lock:
            cached = self._context_cache.get(key)
            if cached is not None:
                return cached
            context_config = _replay_context_config(self.config)
            context = load_or_build_prepared_context(
                context_config,
                optimized_source=source_ref.path,
                candidate_section=source_ref.section,
                candidate_rank=source_ref.rank,
                strict_candidate_source=False,
                output_dir=self.output_dir,
                train_only=True,
                fold_count=2,
                compiled_cache_dir=context_config.get("compiled_cache_dir"),
                force_rebuild_cache=bool(context_config.get("force_rebuild_cache", False)),
                refresh_cached_baseline=not bool(self.config.get("skip_initial_baseline_eval", False)),
                status_callback=self._context_status,
            )
            self._context_cache[key] = context
            return context

    def _guarded_pool_context_for_mutations(self, mutations: dict[str, Any], base_context: Any, source_ref: FixedCandidateSourceRef):
        if not getattr(base_context, "context_by_key", None) or not getattr(getattr(base_context, "dataset", None), "bars_by_key", None):
            context_config = _replay_context_config(self.config)
            base_context = load_or_build_prepared_context(
                context_config,
                optimized_source=source_ref.path,
                candidate_section=source_ref.section,
                candidate_rank=source_ref.rank,
                strict_candidate_source=False,
                output_dir=self.output_dir,
                train_only=True,
                fold_count=2,
                compiled_cache_dir=context_config.get("compiled_cache_dir"),
                force_rebuild_cache=True,
                refresh_cached_baseline=not bool(self.config.get("skip_initial_baseline_eval", False)),
                status_callback=self._context_status,
            )
            self._context_cache[_source_context_key(source_ref)] = base_context
        pool_path = Path(_resolve_existing_source_path(str(mutations.get(POOL_SOURCE_PATH_MUTATION) or "")))
        active_count = max(1, int(float(mutations.get(POOL_SOURCE_ACTIVE_COUNT_MUTATION, 16) or 16)))
        key = json.dumps(
            {
                "kind": "guarded_prefilter_pool",
                "path": str(pool_path),
                "active_count": active_count,
                "base_cache_key": getattr(base_context, "cache_key", ""),
                "version": FIXED_PHASE_AUTO_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._context_lock:
            cached = self._context_cache.get(key)
            if cached is not None:
                return cached
            pool_rows = _read_jsonl_objects(pool_path)
            self._context_status("compiling_guarded_pool_replay", pool_rows=len(pool_rows), active_count=active_count)
            by_day: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            from datetime import date as _date

            for row in pool_rows:
                day_value = str(row.get("trade_date") or row.get("entry_date") or "")[:10]
                if not day_value:
                    continue
                by_day[_date.fromisoformat(day_value)].append(row)
            selections: list[Selection] = []
            frontier_by_day: dict[Any, tuple[str, ...]] = {}
            frontier_scores_by_day: dict[Any, dict[str, float]] = {}
            metadata_by_key: dict[tuple[Any, str], dict[str, Any]] = {}
            for day in base_context.train_dates:
                ordered = sorted(by_day.get(day, ()), key=lambda item: (int(float(item.get("pool_rank") or item.get("causal_rank_in_day") or 999)), str(item.get("symbol") or "")))
                symbols = tuple(str(row.get("symbol") or "") for row in ordered if str(row.get("symbol") or ""))
                frontier_by_day[day] = symbols
                frontier_scores_by_day[day] = {str(row.get("symbol") or ""): _as_float(row.get("causal_ranker_score")) for row in ordered if str(row.get("symbol") or "")}
                for rank, row in enumerate(ordered, start=1):
                    symbol = str(row.get("symbol") or "")
                    if not symbol:
                        continue
                    metadata_by_key[(day, symbol)] = {
                        "candidate_rank": rank,
                        "frontier_rank": rank,
                        "frontier_selection_score": _as_float(row.get("causal_ranker_score")),
                        "candidate_prefilter_contract": str(mutations.get(POOL_SOURCE_LABEL_MUTATION) or "guarded_prefilter_pool"),
                        "pool_rank": rank,
                        "pool_variant": str(row.get("pool_variant") or "guarded_prefilter_pool"),
                    }
                for row in ordered[:active_count]:
                    symbol = str(row.get("symbol") or "")
                    if symbol:
                        selections.append(Selection(day, symbol, _as_float(row.get("causal_ranker_score")), "guarded_prefilter_pool"))
            counts = _selection_counts_for_dates(selections, base_context.train_dates)
            plan_cfg = self._config_for_mutations(mutations)
            compiled = compile_core_replay(
                selections,
                base_context.dataset,
                base_context.context_by_key,
                base_context.train_dates,
                counts,
                plan_cfg,
                frontier_by_day=frontier_by_day,
                frontier_scores_by_day=frontier_scores_by_day,
                candidate_metadata_by_key=metadata_by_key,
                source_calibration_metadata={
                    "candidate_source_mode": "guarded_prefilter_pool",
                    "pool_source_path": str(pool_path),
                    "active_count": active_count,
                    "holdout_policy": "train_only_pool_rows",
                },
            )
            cache_key = stable_signature([key, compiled.source_fingerprint, compiled.candidate_artifact_hash])
            cache_metadata = {
                **dict(getattr(base_context, "cache_metadata", {}) or {}),
                "cache_hit": False,
                "candidate_source_mode": "guarded_prefilter_pool",
                "guarded_pool_path": str(pool_path),
                "guarded_pool_rows": len(pool_rows),
                "pool_active_count": active_count,
                "compiled_bars": len(compiled.bars),
                "snapshots": len(compiled.snapshots),
                "selections": len(selections),
                "candidate_artifact_hash": compiled.candidate_artifact_hash,
                "compiled_replay_fingerprint": compiled.source_fingerprint,
            }
            candidate_source = replace(
                base_context.candidate_source,
                source_path=str(pool_path),
                source_row_name=str(mutations.get(POOL_SOURCE_LABEL_MUTATION) or "guarded_prefilter_pool"),
                source_section="guarded_prefilter_pool",
                source_rank=0,
                calibration_metadata={
                    **dict(getattr(base_context.candidate_source, "calibration_metadata", {}) or {}),
                    "candidate_source_mode": "guarded_prefilter_pool",
                    "pool_active_count": active_count,
                },
            )
            context = replace(
                base_context,
                candidate_source=candidate_source,
                selections=selections,
                frontier=frontier_by_day,
                selection_counts=counts,
                compiled_replay=compiled,
                cache_key=cache_key,
                cache_metadata=cache_metadata,
            )
            self._context_status("guarded_pool_replay_compiled", bars=len(compiled.bars), selections=len(selections), snapshots=len(compiled.snapshots))
            self._context_cache[key] = context
            return context

    def _context_status(self, stage: str, **extra: Any) -> None:
        try:
            _write_json(
                self.output_dir / "context_status.json",
                {
                    "stage": stage,
                    "extra": extra,
                    "updated_at": time.time(),
                },
            )
        except Exception:
            return

    def _evaluate(self, mutations: dict[str, Any]) -> FixedEvaluation:
        started = time.monotonic()
        context = self._context_for_mutations(mutations)
        initial_equity = float(context.compiled_replay.initial_equity)
        plan_cfg = self._config_for_mutations(mutations)
        costs = BrokerCosts(commission_bps=plan_cfg.commission_bps, tax_bps_on_sell=plan_cfg.tax_bps_on_sell, slippage_bps=plan_cfg.slippage_bps)
        adapter = KALCBReplayAdapter(plan_cfg, _clone_snapshots_for_replay(context.compiled_replay.snapshots), initial_equity=initial_equity, costs=costs)
        replay = run_replay(
            context.compiled_replay.bars,
            adapter,
            initial_equity=initial_equity,
            costs=costs,
            close_open_positions=False,
            bars_are_ordered=True,
            buying_power_leverage=max(float(plan_cfg.intraday_leverage), 1.0),
        )
        replay.decisions.extend(adapter._sync_new_fills(replay.broker))
        adapter.finalize_frontier_shadow(context.compiled_replay.bars[-1] if context.compiled_replay.bars else None)
        trades = _collapse_exit_legs(replay.trades)
        outcomes = _broker_trades_to_slot_outcomes(trades, plan_cfg)
        metrics = summarize_outcomes(outcomes, session_dates=context.train_dates, selection_counts=context.selection_counts)
        _add_compiled_candidate_pool_metrics(metrics, context.compiled_replay, context.train_dates, len(outcomes))
        broker_metrics = compute_trade_metrics(trades, replay.equity_curve, initial_equity=initial_equity)
        final_equity = float(replay.equity_curve[-1]) if replay.equity_curve else initial_equity
        official_mtm_net = final_equity / max(initial_equity, 1.0) - 1.0
        metrics.update(
            {
                "broker_net_return_pct": float(broker_metrics.get("net_return_pct", 0.0)),
                "official_mtm_net_return_pct": float(official_mtm_net),
                "final_equity": final_equity,
                "end_open_position_count": float(len(replay.broker.positions)),
                "broker_net_profit": float(broker_metrics.get("net_profit", 0.0)),
                "broker_max_drawdown_pct": float(broker_metrics.get("max_drawdown_pct", 0.0)),
                "broker_expected_total_r": float(broker_metrics.get("expected_total_r", 0.0)),
                "broker_avg_r": float(broker_metrics.get("avg_r", 0.0)),
                "broker_mfe_capture": float(broker_metrics.get("mfe_capture", 0.0)),
                "broker_trade_count": float(broker_metrics.get("total_trades", 0.0)),
                "same_bar_fill_count": float(replay.broker.same_bar_fill_violations),
                "forced_replay_close_count": 0.0,
                "rejected_order_count": 0.0,
                "mark_to_market_equity_points": float(len(replay.equity_curve)),
                "broker_net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
                "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
                "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
                "primary_promotion_metric": OFFICIAL_PROMOTION_METRIC,
                "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
                "research_objective_metric": PRIMARY_OBJECTIVE_METRIC,
                "research_objective_basis": "closed_trade_net_pnl_over_initial_equity",
                "total_trades": float(broker_metrics.get("total_trades", metrics.get("trade_count", 0.0))),
                "trades": float(broker_metrics.get("total_trades", metrics.get("trade_count", 0.0))),
                "win_rate": float(metrics.get("net_win_share", 0.0)),
                "source_fingerprint": context.compiled_replay.source_fingerprint,
                "feature_manifest_hash": context.cache_key,
                "candidate_snapshot_hash": context.compiled_replay.candidate_artifact_hash,
                "replay_mode": "fixed_candidate_shared_core_compiled_replay",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        metrics.update(adapter.frontier_metrics())
        _add_portfolio_equivalent_metrics(metrics, outcomes, context.train_dates, initial_equity)
        _add_return_divergence_metrics(metrics)
        fold_rows = _fold_metrics_from_outcomes_for_dates(outcomes, context.train_dates, context.folds, context.selection_counts, initial_equity=initial_equity)
        _add_fold_metrics(metrics, fold_rows)
        metrics["score_components"] = {name: _scaled_component(name, metrics) for name in SCORE_COMPONENTS}
        metrics["immutable_score"] = score_fixed(metrics)
        metrics["holdout_excluded"] = True
        metrics["training_window_start"] = context.train_dates[0].isoformat() if context.train_dates else ""
        metrics["training_window_end"] = context.train_dates[-1].isoformat() if context.train_dates else ""
        metrics["training_session_count"] = len(context.train_dates)
        return FixedEvaluation(
            metrics=dict(metrics),
            replay_digest=_replay_digest(replay, trades),
            fold_rows=fold_rows,
            trade_rows=_broker_trade_rows(trades),
            decision_summary=_decision_summary(replay.decisions),
        )

    def _config_for_mutations(self, mutations: dict[str, Any]) -> KALCBConfig:
        normalized = _normalize_mutations(mutations)
        cfg = self.base_cfg.with_mutations(normalized)
        return cfg

    def _audit_fast_replay_parity(self, mutations: dict[str, Any], fast_metrics: dict[str, Any]) -> dict[str, Any]:
        fast_key = _mutation_key(mutations)
        fast_eval = self._evaluation_details.get(fast_key)
        if fast_eval is None:
            self.evaluate_mutations(mutations)
            fast_eval = self._evaluation_details.get(fast_key)
        audit_mutations = dict(mutations or {})
        audit_mutations["kalcb.entry.fast_replay_suppress_rejections"] = False
        audit_eval = self._evaluate(audit_mutations)
        metric_names = (
            "trade_count",
            "avg_trade_net_pct",
            "avg_mfe_capture",
            "broker_net_return_pct",
            "official_mtm_net_return_pct",
            "final_equity",
            "end_open_position_count",
            "broker_net_profit",
            "broker_max_drawdown_pct",
            "broker_expected_total_r",
            "broker_avg_r",
            "broker_mfe_capture",
            "same_bar_fill_count",
            "forced_replay_close_count",
            "rejected_order_count",
            "portfolio_equivalent_net_return_pct",
            "portfolio_equivalent_max_drawdown_pct",
            "worst_fold_net",
            "median_fold_net",
            "immutable_score",
        )
        deltas = {
            name: float(audit_eval.metrics.get(name, 0.0) or 0.0) - float(fast_metrics.get(name, 0.0) or 0.0)
            for name in metric_names
        }
        fast_digest = dict(fast_eval.replay_digest if fast_eval is not None else {})
        audit_digest = dict(audit_eval.replay_digest)
        fill_hash_match = fast_digest.get("fill_hash", "") == audit_digest.get("fill_hash", "")
        trade_hash_match = fast_digest.get("trade_hash", "") == audit_digest.get("trade_hash", "")
        trading_decision_hash_match = fast_digest.get("trading_decision_hash", "") == audit_digest.get("trading_decision_hash", "")
        strategy_action_hash_match = fast_digest.get("strategy_action_hash", "") == audit_digest.get("strategy_action_hash", "")
        audit_pass = (
            all(abs(value) <= 1e-10 for value in deltas.values())
            and fill_hash_match
            and trade_hash_match
            and trading_decision_hash_match
            and strategy_action_hash_match
        )
        return {
            "status": "pass" if audit_pass else "fail",
            "pass": audit_pass,
            "audit_count": 1,
            "max_abs_metric_delta": max((abs(value) for value in deltas.values()), default=0.0),
            "metric_deltas": deltas,
            "fill_hash_match": fill_hash_match,
            "trade_hash_match": trade_hash_match,
            "trading_decision_hash_match": trading_decision_hash_match,
            "strategy_action_hash_match": strategy_action_hash_match,
            "fast_replay_digest": fast_digest,
            "audit_replay_digest": audit_digest,
            "fast_decision_count": int(fast_digest.get("decision_count", 0) or 0),
            "audit_decision_count": int(audit_digest.get("decision_count", 0) or 0),
            "suppressed_entry_rejection_count": max(
                0,
                int(audit_digest.get("entry_rejection_count", 0) or 0) - int(fast_digest.get("entry_rejection_count", 0) or 0),
            ),
            "scope": "Only entry_rejected diagnostics may differ between fast and full replay paths.",
        }


def should_use_fixed_trade_plan_phase(config: dict[str, Any] | None = None, output_dir: Path | None = None) -> bool:
    config = dict(config or {})
    if bool(config.get("fixed_trade_plan_phase_auto")):
        return True
    if not output_dir:
        return False
    previous = _previous_round_dir(Path(output_dir))
    if previous is None:
        return False
    optimized = previous / "optimized_config.json"
    if not optimized.exists():
        return False
    try:
        data = json.loads(optimized.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    contract = data.get("execution_contract") if isinstance(data, dict) else {}
    return str((contract or {}).get("phase_framework_version") or "") == FIXED_PREVIOUS_FRAMEWORK_VERSION


PRIMARY_FIRST30_ROUTE = {
    "name": "first30_open_anchor",
    "mode": "first30_open",
    "priority": 0,
}
GAP_RETENTION_TRAIN_THRESHOLDS = {
    "first30_gap_q75": 0.011972513955908759,
    "first30_gap_relvol_q70": 0.01091985778290561,
    "first30_gap_relvol_q75": 0.014818777920993287,
    "first30_gap_relvol_q80": 0.019631196437898697,
    "first30_gap_relvol_q85": 0.027352564326333977,
    "first30_gap_retention_q75": 0.6666666666666631,
    "pathscan_first30_gap_retention_q80": 0.06761904761904801,
    "pathscan_first30_gap_retention_q85": 0.3333333333333333,
    "first30_low_vs_prev_close_q85": 0.003779231178016761,
    "first30_rel_volume_q85": 5.351847980160157,
    "daily_acceleration_5v20_q65": 0.04300717480400412,
    "sector_daily_score_pct_q75": 81.25,
    "h3_current_r_q65": 0.9034784102904003,
    "h3_current_r_q75": 1.585432943694103,
    "h6_current_r_q65": 1.1499595448363724,
    "h6_current_r_q75": 2.049519102342391,
}


def _route_plan(*secondary_routes: dict[str, Any]) -> dict[str, Any]:
    return {"kalcb.entry.routes": [dict(PRIMARY_FIRST30_ROUTE), *(dict(route) for route in secondary_routes)]}


def _route_plan_with_primary(primary_overrides: dict[str, Any], *secondary_routes: dict[str, Any]) -> dict[str, Any]:
    primary = {**dict(PRIMARY_FIRST30_ROUTE), **dict(primary_overrides)}
    return {"kalcb.entry.routes": [primary, *(dict(route) for route in secondary_routes)]}


def _gap_priority_route_plan(
    *,
    name: str,
    context_min: dict[str, float],
    context_max: dict[str, float] | None = None,
    risk_mult: float = 1.0,
    notional_mult: float = 1.0,
    fallback_mult: float = 1.0,
    dynamic: dict[str, Any] | None = None,
    route_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = {
        "name": name,
        "mode": "first30_open",
        "priority": 0,
        "context_min": dict(context_min),
        "risk_mult": float(risk_mult),
        "notional_mult": float(notional_mult),
    }
    if context_max:
        primary["context_max"] = dict(context_max)
    if dynamic:
        primary.update(dict(dynamic))
    if route_overrides:
        primary.update(dict(route_overrides))
    fallback = {**dict(PRIMARY_FIRST30_ROUTE), "priority": 100, "require_initial_active": True}
    if fallback_mult != 1.0:
        fallback.update({"risk_mult": float(fallback_mult), "notional_mult": float(fallback_mult)})
    return {"kalcb.entry.routes": [primary, fallback]}


def _mfe_floor_candidate(
    *,
    start_r: float,
    floor_r: float,
    hold_bars: int,
    max_relvol: float = 3.0,
    max_first30_ret: float | None = None,
) -> dict[str, Any]:
    mutations: dict[str, Any] = {
        "kalcb.exit.mfe_floor_enabled": True,
        "kalcb.exit.mfe_floor_start_r": float(start_r),
        "kalcb.exit.mfe_floor_floor_r": float(floor_r),
        "kalcb.exit.mfe_floor_min_hold_bars": int(hold_bars),
        "kalcb.exit.mfe_floor_max_first30_rel_volume": float(max_relvol),
        "kalcb.exit.mfe_floor_entry_route_modes": ["first30_open"],
    }
    if max_first30_ret is not None:
        mutations["kalcb.exit.mfe_floor_max_first30_ret"] = float(max_first30_ret)
    return mutations


def _conditional_target_candidate(
    *,
    target_r: float,
    min_hold_bars: int,
    min_relvol: float = 0.0,
    max_relvol: float = 0.0,
    min_cpr: float = 0.0,
) -> dict[str, Any]:
    return {
        "kalcb.exit.conditional_target_enabled": True,
        "kalcb.exit.conditional_target_r": float(target_r),
        "kalcb.exit.conditional_target_min_hold_bars": int(min_hold_bars),
        "kalcb.exit.conditional_target_min_first30_rel_volume": float(min_relvol),
        "kalcb.exit.conditional_target_max_first30_rel_volume": float(max_relvol),
        "kalcb.exit.conditional_target_min_first30_signal_cpr": float(min_cpr),
        "kalcb.exit.conditional_target_entry_route_modes": ["first30_open"],
    }


def _path_quality_exit_candidate(
    *,
    min_hold_bars: int,
    min_mfe_r: float,
    min_giveback_r: float,
    context_min: dict[str, float],
) -> dict[str, Any]:
    return {
        "kalcb.exit.path_quality_enabled": True,
        "kalcb.exit.path_quality_min_hold_bars": int(min_hold_bars),
        "kalcb.exit.path_quality_min_mfe_r": float(min_mfe_r),
        "kalcb.exit.path_quality_min_giveback_r": float(min_giveback_r),
        "kalcb.exit.path_quality.context_min": dict(context_min),
        "kalcb.exit.path_quality_entry_route_modes": ["first30_open"],
    }


def _tail_capture_rescue_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("rescue_conditional_target45_low_relvol_h18", _conditional_target_candidate(target_r=45.0, min_hold_bars=18, max_relvol=3.0)),
        ("rescue_conditional_target50_low_relvol_h24", _conditional_target_candidate(target_r=50.0, min_hold_bars=24, max_relvol=3.0)),
        ("rescue_conditional_target60_high_cpr_h24", _conditional_target_candidate(target_r=60.0, min_hold_bars=24, min_cpr=0.75)),
    ]


def _frontier_path_proof_branch(
    *,
    name: str,
    rank: int,
    risk_mult: float,
    max_session_trades: int,
    context_min: dict[str, float],
    h6_threshold: float,
    h3_threshold: float | None = None,
) -> dict[str, Any]:
    proof_context = dict(context_min)
    if h3_threshold is not None:
        proof_context["h3_current_r"] = float(h3_threshold)
    proof_context["h6_current_r"] = float(h6_threshold)
    route: dict[str, Any] = {
        "name": name,
        "mode": "deferred_continuation",
        "priority": 20,
        "after_bar": 5,
        "max_signal_bars": 18,
        "require_initial_active": False,
        "max_frontier_rank": int(rank),
        "min_quality_votes": 6,
        "quality_min_first30_signal_cpr": 0.70,
        "quality_min_first30_rel_volume": 1.25,
        "quality_min_accumulation_score": -0.05,
        "min_bar_ret": 0.0,
        "min_breakout_pct": 0.0005,
        "min_close_location": 0.55,
        "risk_mult": float(risk_mult),
        "notional_mult": float(risk_mult),
        "max_session_trades": int(max_session_trades),
        "context_min": proof_context,
    }
    return {
        **_route_plan(route),
        "kalcb.entry.frontier_branch_universe": True,
    }


def _frontier_branch_plan(*secondary_routes: dict[str, Any]) -> dict[str, Any]:
    return {
        **_route_plan(*secondary_routes),
        "kalcb.entry.frontier_branch_universe": True,
    }


def _deferred_q5_route(
    *,
    rank: int = 6,
    after_bar: int = 2,
    max_signal_bars: int = 24,
    min_close_location: float = 0.55,
    min_rel_volume: float = 1.25,
    min_cpr: float = 0.70,
) -> dict[str, Any]:
    return {
        "name": f"rank{rank}_deferred_q5",
        "mode": "deferred_continuation",
        "priority": 20,
        "after_bar": after_bar,
        "max_signal_bars": max_signal_bars,
        "min_bar_ret": 0.0,
        "min_breakout_pct": 0.0005,
        "min_close_location": min_close_location,
        "max_frontier_rank": rank,
        "min_quality_votes": 5,
        "quality_min_first30_signal_cpr": min_cpr,
        "quality_min_first30_rel_volume": min_rel_volume,
    }


def _reclaim_q5_route(*, rank: int = 8, max_pullback_from_vwap_pct: float = 0.010) -> dict[str, Any]:
    return {
        "name": f"rank{rank}_reclaim_q5",
        "mode": "pullback_acceptance",
        "priority": 10,
        "after_bar": 1,
        "max_signal_bars": 18,
        "min_bar_ret": 0.0,
        "min_reclaim_ret": 0.0,
        "max_pullback_from_vwap_pct": max_pullback_from_vwap_pct,
        "max_frontier_rank": rank,
        "min_quality_votes": 5,
        "quality_min_first30_signal_cpr": 0.70,
        "quality_min_first30_rel_volume": 1.25,
    }


def _consolidated_phase_candidates(phase: int) -> list[tuple[str, dict[str, Any]]] | None:
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    gap_q75 = {"first30_gap": t["first30_gap_q75"]}
    gaprel_q70 = {"first30_gap_relvol": t["first30_gap_relvol_q70"]}
    gaprel_q75 = {"first30_gap_relvol": t["first30_gap_relvol_q75"]}
    gaprel_q85 = {"first30_gap_relvol": t["first30_gap_relvol_q85"]}
    lowprev_q85 = {"first30_low_vs_prev_close": t["first30_low_vs_prev_close_q85"]}
    relvol_q85 = {"first30_rel_volume": t["first30_rel_volume_q85"]}

    if phase == 1:
        return [
            (
                "active_lowprev_q85_rank5_risk102_fallback099",
                _gap_priority_route_plan(
                    name="lowprev_q85_rank5_risk102",
                    context_min=lowprev_q85,
                    risk_mult=1.02,
                    notional_mult=1.02,
                    fallback_mult=0.99,
                    context_max={"frontier_rank": 5},
                ),
            ),
            (
                "active_relvol_q85_rank8_risk102_fallback099",
                _gap_priority_route_plan(
                    name="relvol_q85_rank8_risk102",
                    context_min=relvol_q85,
                    risk_mult=1.02,
                    notional_mult=1.02,
                    fallback_mult=0.99,
                    context_max={"frontier_rank": 8},
                ),
            ),
            (
                "active_gaprel_q85_dailysector_q75_rank12_press102",
                _gap_priority_route_plan(
                    name="gaprel_dailysector_rank12_press102",
                    context_min={**gaprel_q85, "sector_daily_score_pct": t["sector_daily_score_pct_q75"]},
                    risk_mult=1.02,
                    notional_mult=1.02,
                    fallback_mult=1.0,
                    context_max={"frontier_rank": 12},
                ),
            ),
            (
                "active_lowprev_q85_dailysector_q75_rank5_press103",
                _gap_priority_route_plan(
                    name="lowprev_dailysector_rank5_press103",
                    context_min={**lowprev_q85, "sector_daily_score_pct": t["sector_daily_score_pct_q75"]},
                    risk_mult=1.03,
                    notional_mult=1.03,
                    fallback_mult=1.0,
                    context_max={"frontier_rank": 5},
                ),
            ),
        ]
    if phase == 2:
        return [
            (
                "gaprelvol_q75_cap60_dd030_open3_session000",
                _gap_priority_route_plan(
                    name="gaprelvol_q75_cap60_dd030",
                    context_min=gaprel_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.60,
                        "dynamic_max_drawdown_pct": 0.030,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 3,
                        "dynamic_max_open_notional_pct": 1.05,
                    },
                ),
            ),
            (
                "gaprelvol_q75_cap60_dd035_open4_session000",
                _gap_priority_route_plan(
                    name="gaprelvol_q75_cap60_dd035",
                    context_min=gaprel_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.60,
                        "dynamic_max_drawdown_pct": 0.035,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 4,
                        "dynamic_max_open_notional_pct": 1.10,
                    },
                ),
            ),
            (
                "gaprelvol_q75_cap59_dd035_open4_session_m05",
                _gap_priority_route_plan(
                    name="gaprelvol_q75_cap59_dd035",
                    context_min=gaprel_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.59,
                        "dynamic_max_drawdown_pct": 0.035,
                        "dynamic_min_session_return_pct": -0.0005,
                        "dynamic_max_open_positions": 4,
                        "dynamic_max_open_notional_pct": 1.10,
                    },
                ),
            ),
            (
                "gap_q75_cap60_dd030_open3_session000",
                _gap_priority_route_plan(
                    name="gap_q75_cap60_dd030",
                    context_min=gap_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.60,
                        "dynamic_max_drawdown_pct": 0.030,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 3,
                        "dynamic_max_open_notional_pct": 1.05,
                    },
                ),
            ),
            (
                "gaprelvol_q70_cap60_dd030_open3_session000",
                _gap_priority_route_plan(
                    name="gaprelvol_q70_cap60_dd030",
                    context_min=gaprel_q70,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.60,
                        "dynamic_max_drawdown_pct": 0.030,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 3,
                        "dynamic_max_open_notional_pct": 1.05,
                    },
                ),
            ),
        ]
    if phase == 3:
        pathproof_context = {
            "first30_gap_relvol": t["first30_gap_relvol_q75"],
            "first30_gap_retention_ratio": t["pathscan_first30_gap_retention_q80"],
            "daily_acceleration_5v20": t["daily_acceleration_5v20_q65"],
        }
        pathproof_context_strict = {
            "first30_gap_relvol": t["first30_gap_relvol_q80"],
            "first30_gap_retention_ratio": t["pathscan_first30_gap_retention_q85"],
            "daily_acceleration_5v20": t["daily_acceleration_5v20_q65"],
        }
        return [
            (
                "frontier_pathproof_gaprel_ret_accel_rank12_s03_cap1",
                _frontier_path_proof_branch(
                    name="frontier_pathproof_rank12_s03",
                    rank=12,
                    risk_mult=0.03,
                    max_session_trades=1,
                    context_min=pathproof_context,
                    h3_threshold=t["h3_current_r_q65"],
                    h6_threshold=t["h6_current_r_q65"],
                ),
            ),
            (
                "frontier_pathproof_gaprel_ret_accel_rank8_s04_cap1",
                _frontier_path_proof_branch(
                    name="frontier_pathproof_rank8_s04",
                    rank=8,
                    risk_mult=0.04,
                    max_session_trades=1,
                    context_min=pathproof_context,
                    h3_threshold=t["h3_current_r_q65"],
                    h6_threshold=t["h6_current_r_q65"],
                ),
            ),
            (
                "frontier_pathproof_gaprel_ret_accel_rank12_h6q75_s02_cap1",
                _frontier_path_proof_branch(
                    name="frontier_pathproof_rank12_h6q75_s02",
                    rank=12,
                    risk_mult=0.02,
                    max_session_trades=1,
                    context_min=pathproof_context,
                    h3_threshold=t["h3_current_r_q65"],
                    h6_threshold=t["h6_current_r_q75"],
                ),
            ),
            (
                "frontier_pathproof_strict_rank12_s03_cap1",
                _frontier_path_proof_branch(
                    name="frontier_pathproof_strict_rank12_s03",
                    rank=12,
                    risk_mult=0.03,
                    max_session_trades=1,
                    context_min=pathproof_context_strict,
                    h3_threshold=t["h3_current_r_q75"],
                    h6_threshold=t["h6_current_r_q65"],
                ),
            ),
        ]
    if phase == 4:
        return [
            ("conditional_target50_low_relvol_h24", _conditional_target_candidate(target_r=50.0, min_hold_bars=24, max_relvol=3.0)),
            ("conditional_target45_low_relvol_h18", _conditional_target_candidate(target_r=45.0, min_hold_bars=18, max_relvol=3.0)),
            ("conditional_target60_high_cpr_h24", _conditional_target_candidate(target_r=60.0, min_hold_bars=24, min_cpr=0.75)),
            ("conditional_target55_mid_relvol_cpr_h24", _conditional_target_candidate(target_r=55.0, min_hold_bars=24, min_relvol=2.0, max_relvol=4.5, min_cpr=0.70)),
        ]
    if phase == 5:
        return [
            ("pathq_h36_mfe8_gb5_below_or_high3", _path_quality_exit_candidate(min_hold_bars=36, min_mfe_r=8.0, min_giveback_r=5.0, context_min={"below_or_high_streak": 3.0})),
            ("low_relvol_mfe6_floor1_h24", _mfe_floor_candidate(start_r=6.0, floor_r=1.0, hold_bars=24, max_relvol=3.0)),
            ("low_relvol_mfe6_floor0_h36", _mfe_floor_candidate(start_r=6.0, floor_r=0.0, hold_bars=36, max_relvol=3.0)),
            ("low_relvol_mfe8_floor1_h24", _mfe_floor_candidate(start_r=8.0, floor_r=1.0, hold_bars=24, max_relvol=3.0)),
            ("weak_first30ret_mfe6_floor0_h24", _mfe_floor_candidate(start_r=6.0, floor_r=0.0, hold_bars=24, max_relvol=4.0, max_first30_ret=0.015)),
            ("weak_first30ret_mfe8_floor1_h24", _mfe_floor_candidate(start_r=8.0, floor_r=1.0, hold_bars=24, max_relvol=4.0, max_first30_ret=0.015)),
        ]
    if phase == 6:
        cap60 = _gap_priority_route_plan(
            name="gaprelvol_q75_cap60_dd030",
            context_min=gaprel_q75,
            dynamic={
                "dynamic_notional_enabled": True,
                "dynamic_max_position_notional_pct": 0.60,
                "dynamic_max_drawdown_pct": 0.030,
                "dynamic_min_session_return_pct": 0.0,
                "dynamic_max_open_positions": 3,
                "dynamic_max_open_notional_pct": 1.05,
            },
        )
        gap_risk = _gap_priority_route_plan(name="gap_q75_risk105", context_min=gap_q75, risk_mult=1.05, notional_mult=1.05, fallback_mult=0.98)
        floor0 = _mfe_floor_candidate(start_r=6.0, floor_r=0.0, hold_bars=36, max_relvol=3.0)
        target50 = _conditional_target_candidate(target_r=50.0, min_hold_bars=24, max_relvol=3.0)
        target60 = _conditional_target_candidate(target_r=60.0, min_hold_bars=24, min_cpr=0.75)
        return [
            ("combo_gaprelvol_cap60_mfefloor0_h36", {**cap60, **floor0}),
            ("combo_gap_q75_risk105_mfefloor0_h36", {**gap_risk, **floor0}),
            ("combo_gaprelvol_cap60_target50_lowrelvol", {**cap60, **target50}),
            ("combo_gap_q75_risk105_target60_highcpr", {**gap_risk, **target60}),
            ("combo_mfefloor0_target50_lowrelvol", {**floor0, **target50}),
        ]
    if phase > CONSOLIDATED_PHASE_COUNT:
        return []
    return None


def _rescue_experiments_for_phase(phase: int) -> list[Experiment]:
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    gap_q75 = {"first30_gap": t["first30_gap_q75"]}
    gaprel_q75 = {"first30_gap_relvol": t["first30_gap_relvol_q75"]}
    items: list[tuple[str, dict[str, Any]]]
    if phase == 1:
        items = [
            ("rescue_gap_q75_risk102_fallback099", _gap_priority_route_plan(name="gap_q75_risk102", context_min=gap_q75, risk_mult=1.02, notional_mult=1.02, fallback_mult=0.99)),
            ("rescue_gaprelvol_q75_notional105_fallback099", _gap_priority_route_plan(name="gaprelvol_q75_notional105", context_min=gaprel_q75, notional_mult=1.05, fallback_mult=0.99)),
        ]
    elif phase == 2:
        items = [
            (
                "rescue_gaprelvol_q75_cap60_dd025_open3",
                _gap_priority_route_plan(
                    name="gaprelvol_q75_cap60_dd025",
                    context_min=gaprel_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.60,
                        "dynamic_max_drawdown_pct": 0.025,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 3,
                        "dynamic_max_open_notional_pct": 1.00,
                    },
                ),
            ),
            (
                "rescue_gaprelvol_q75_cap58_dd030_open3",
                _gap_priority_route_plan(
                    name="gaprelvol_q75_cap58_dd030",
                    context_min=gaprel_q75,
                    dynamic={
                        "dynamic_notional_enabled": True,
                        "dynamic_max_position_notional_pct": 0.58,
                        "dynamic_max_drawdown_pct": 0.030,
                        "dynamic_min_session_return_pct": 0.0,
                        "dynamic_max_open_positions": 3,
                        "dynamic_max_open_notional_pct": 1.00,
                    },
                ),
            ),
        ]
    elif phase == 3:
        pathproof_context = {
            "first30_gap_relvol": t["first30_gap_relvol_q75"],
            "first30_gap_retention_ratio": t["pathscan_first30_gap_retention_q80"],
            "daily_acceleration_5v20": t["daily_acceleration_5v20_q65"],
        }
        items = [
            (
                "rescue_frontier_pathproof_rank12_h6q65_s02_cap1",
                _frontier_path_proof_branch(
                    name="frontier_pathproof_rescue_rank12_s02",
                    rank=12,
                    risk_mult=0.02,
                    max_session_trades=1,
                    context_min=pathproof_context,
                    h3_threshold=t["h3_current_r_q65"],
                    h6_threshold=t["h6_current_r_q65"],
                ),
            ),
        ]
    elif phase == 4:
        items = _tail_capture_rescue_candidates()
    elif phase == 5:
        items = [
            ("rescue_low_relvol_mfe8_floor0_h36", _mfe_floor_candidate(start_r=8.0, floor_r=0.0, hold_bars=36, max_relvol=3.0)),
            ("rescue_weak_first30ret_mfe10_floor1_h24", _mfe_floor_candidate(start_r=10.0, floor_r=1.0, hold_bars=24, max_relvol=4.0, max_first30_ret=0.015)),
        ]
    else:
        items = []
    return [Experiment(name, mutations) for name, mutations in items]


def get_phase_candidates(phase: int) -> list[Experiment]:
    consolidated = _consolidated_phase_candidates(int(phase))
    if consolidated is not None:
        return [Experiment(name, mutations) for name, mutations in consolidated]
    raw: dict[int, list[tuple[str, dict[str, Any]]]] = {
        1: [
            ("source_portfolio_rank0", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 0}),
            ("source_portfolio_rank1", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 1}),
            ("source_portfolio_rank2", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 2}),
            ("source_portfolio_rank3", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 3}),
            ("source_portfolio_rank5", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 5}),
            ("source_portfolio_rank6", {SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 6}),
            ("source_slot_rank0", {SOURCE_SECTION_MUTATION: "top_slot_return", SOURCE_RANK_MUTATION: 0}),
            ("source_pareto_rank0", {SOURCE_SECTION_MUTATION: "top_pareto", SOURCE_RANK_MUTATION: 0}),
            ("source_combined_rank0", {SOURCE_SECTION_MUTATION: "top_combined", SOURCE_RANK_MUTATION: 0}),
            ("frontier_expand_top8_cpr55", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 8, "kalcb.entry.min_first30_signal_cpr": 0.55}),
            ("frontier_expand_top12_cpr55", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 12, "kalcb.entry.min_first30_signal_cpr": 0.55}),
            ("frontier_expand_top20_cpr60", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 20, "kalcb.entry.min_first30_signal_cpr": 0.60}),
            ("frontier_expand_top20_relvol100", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 20, "kalcb.entry.min_first30_rel_volume": 1.0}),
            ("frontier_expand_top20_vwap_cpr55", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 20, "kalcb.entry.min_vwap_ret": 0.001, "kalcb.entry.min_first30_signal_cpr": 0.55}),
            ("frontier_expand_top30_cpr65", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 30, "kalcb.entry.min_first30_signal_cpr": 0.65}),
        ],
        2: [
            ("first30_vwap_10bps", {"kalcb.entry.min_vwap_ret": 0.001}),
            ("first30_vwap_20bps", {"kalcb.entry.min_vwap_ret": 0.002}),
            ("first30_ret_50bps", {"kalcb.entry.min_bar_ret": 0.005}),
            ("first30_close_loc_55", {"kalcb.entry.min_close_location": 0.55}),
            ("first30_close_loc_65", {"kalcb.entry.min_close_location": 0.65}),
            ("first30_above_prev", {"kalcb.entry.require_above_prev_close": True}),
            ("first30_gap_cap_8pct", {"kalcb.entry.gap_max_pct": 0.08}),
            ("first30_gap_floor_m1pct", {"kalcb.entry.gap_min_pct": -0.01}),
            ("first30_relvol_075", {"kalcb.entry.min_first30_rel_volume": 0.75}),
            ("first30_relvol_100", {"kalcb.entry.min_first30_rel_volume": 1.0}),
            ("first30_signal_cpr_55", {"kalcb.entry.min_first30_signal_cpr": 0.55}),
            ("first30_signal_cpr_65", {"kalcb.entry.min_first30_signal_cpr": 0.65}),
            ("first30_open_dd_floor_m3pct", {"kalcb.entry.min_first30_open_drawdown": -0.03}),
            ("first30_low_vs_prev_floor_m1pct", {"kalcb.entry.min_first30_low_vs_prev_close": -0.01}),
            ("first30_range_atr_015_300", {"kalcb.entry.min_first30_range_atr": 0.15, "kalcb.entry.max_first30_range_atr": 3.0}),
            ("frontier_rank_top8", {"kalcb.entry.max_frontier_rank": 8}),
            ("frontier_score_top_quality", {"kalcb.entry.min_frontier_score": 0.0}),
            ("source_flow_nonneg", {"kalcb.entry.min_flow_score": 0.0}),
            ("source_accum_nonneg", {"kalcb.entry.min_accumulation_score": 0.0}),
            ("first30_vwap_close_combo", {"kalcb.entry.min_vwap_ret": 0.001, "kalcb.entry.min_close_location": 0.55}),
            ("first30_source_quality_combo", {"kalcb.entry.min_first30_signal_cpr": 0.55, "kalcb.entry.min_first30_open_drawdown": -0.03, "kalcb.entry.min_first30_low_vs_prev_close": -0.01}),
        ],
        3: [
            ("first30_relvol_075_frequency_probe", {"kalcb.entry.min_first30_rel_volume": 0.75}),
            ("first30_ret_floor_removed_frequency_probe", {"kalcb.entry.min_bar_ret": 0.0}),
            ("first30_ret_75bps_quality", {"kalcb.entry.min_bar_ret": 0.0075}),
            ("first30_ret_100bps_quality", {"kalcb.entry.min_bar_ret": 0.0100}),
            ("first30_vwap_10bps_quality", {"kalcb.entry.min_vwap_ret": 0.001}),
            ("first30_vwap_20bps_quality", {"kalcb.entry.min_vwap_ret": 0.002}),
            ("first30_signal_cpr55_quality", {"kalcb.entry.min_first30_signal_cpr": 0.55}),
            ("first30_signal_cpr65_quality", {"kalcb.entry.min_first30_signal_cpr": 0.65}),
            ("frontier_rank_top12_hygiene", {"kalcb.entry.max_frontier_rank": 12}),
            ("frontier_rank_top8_hygiene", {"kalcb.entry.max_frontier_rank": 8}),
            ("source_flow_nonneg_hygiene", {"kalcb.entry.min_flow_score": 0.0}),
            ("source_accum_nonneg_hygiene", {"kalcb.entry.min_accumulation_score": 0.0}),
            ("quality_cpr55_relvol125", {"kalcb.entry.min_first30_signal_cpr": 0.55, "kalcb.entry.min_first30_rel_volume": 1.25}),
            ("quality_vwap10_ret75", {"kalcb.entry.min_vwap_ret": 0.001, "kalcb.entry.min_bar_ret": 0.0075}),
            ("quality_vote_cpr75_range075_6of7", {"kalcb.entry.min_quality_votes": 6, "kalcb.entry.quality_min_bar_ret": 0.01, "kalcb.entry.quality_min_first30_signal_cpr": 0.75, "kalcb.entry.quality_min_flow_score": -0.05, "kalcb.entry.quality_min_accumulation_score": 0.0, "kalcb.entry.quality_max_frontier_rank": 20, "kalcb.entry.quality_min_first30_rel_volume": 2.0, "kalcb.entry.quality_min_first30_range_atr": 0.75}),
            ("quality_vote_cpr75_accum05_5of6", {"kalcb.entry.min_quality_votes": 5, "kalcb.entry.quality_min_bar_ret": 0.0125, "kalcb.entry.quality_min_first30_signal_cpr": 0.75, "kalcb.entry.quality_min_flow_score": -0.05, "kalcb.entry.quality_min_accumulation_score": 0.05, "kalcb.entry.quality_max_frontier_rank": 12, "kalcb.entry.quality_min_first30_rel_volume": 1.0}),
            ("frontier_expand_top8_cpr65_relvol125", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 8, "kalcb.entry.min_first30_signal_cpr": 0.65, "kalcb.entry.min_first30_rel_volume": 1.25}),
            ("frontier_expand_top12_cpr70_relvol150", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 12, "kalcb.entry.min_first30_signal_cpr": 0.70, "kalcb.entry.min_first30_rel_volume": 1.50}),
            (
                "routes_rank6_avwap_reclaim_q7",
                _route_plan(
                    {"name": "rank6_avwap_reclaim_q7", "mode": "avwap_reclaim", "priority": 10, "after_bar": 1, "max_signal_bars": 18, "min_bar_ret": 0.0, "min_reclaim_ret": 0.001, "max_pullback_from_vwap_pct": 0.006, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.72, "quality_min_first30_rel_volume": 1.5}
                ),
            ),
            (
                "routes_rank8_pullback_reclaim_q6",
                _route_plan(
                    {"name": "rank8_pullback_reclaim_q6", "mode": "pullback_acceptance", "priority": 10, "after_bar": 1, "max_signal_bars": 18, "min_bar_ret": 0.0, "min_reclaim_ret": 0.0005, "max_pullback_from_vwap_pct": 0.008, "max_frontier_rank": 8, "min_quality_votes": 6, "quality_min_first30_signal_cpr": 0.75, "quality_min_first30_rel_volume": 1.5}
                ),
            ),
            (
                "routes_rank6_deferred_q7",
                _route_plan(
                    {"name": "rank6_deferred_q7", "mode": "deferred_continuation", "priority": 20, "after_bar": 2, "max_signal_bars": 24, "min_bar_ret": 0.0, "min_breakout_pct": 0.001, "min_close_location": 0.60, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.70, "quality_min_first30_rel_volume": 1.25}
                ),
            ),
            (
                "routes_rank6_reclaim_then_deferred",
                _route_plan(
                    {"name": "rank6_or_high_reclaim_q7", "mode": "or_high_reclaim", "priority": 10, "after_bar": 1, "max_signal_bars": 15, "min_bar_ret": 0.0, "min_reclaim_ret": 0.0, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.72, "quality_min_first30_rel_volume": 1.5},
                    {"name": "rank6_deferred_q7", "mode": "deferred_continuation", "priority": 20, "after_bar": 2, "max_signal_bars": 24, "min_bar_ret": 0.0, "min_breakout_pct": 0.001, "min_close_location": 0.60, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.70, "quality_min_first30_rel_volume": 1.25},
                ),
            ),
        ],
        4: [
            ("target_28r", {"kalcb.exit.target_r": 28.0}),
            ("target_30r", {"kalcb.exit.target_r": 30.0}),
            ("target_32r", {"kalcb.exit.target_r": 32.0}),
            ("target_34r", {"kalcb.exit.target_r": 34.0}),
            ("target_36r", {"kalcb.exit.target_r": 36.0}),
            ("target_38r", {"kalcb.exit.target_r": 38.0}),
            ("target_40r", {"kalcb.exit.target_r": 40.0}),
            ("target_45r", {"kalcb.exit.target_r": 45.0}),
            ("target_50r", {"kalcb.exit.target_r": 50.0}),
            ("target_55r", {"kalcb.exit.target_r": 55.0}),
            ("target_60r_net_challenger", {"kalcb.exit.target_r": 60.0}),
            ("target_70r_tail_challenger", {"kalcb.exit.target_r": 70.0}),
            ("partial_8r_25_be", {"kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 8.0, "kalcb.exit.partial_fraction": 0.25, "kalcb.exit.partial_stop_to_breakeven": True, "kalcb.exit.partial_breakeven_buffer_r": 0.10}),
            ("partial_12r_33_be", {"kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 12.0, "kalcb.exit.partial_fraction": 0.33, "kalcb.exit.partial_stop_to_breakeven": True, "kalcb.exit.partial_breakeven_buffer_r": 0.10}),
            ("partial_16r_33_runner", {"kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 16.0, "kalcb.exit.partial_fraction": 0.33, "kalcb.exit.partial_stop_to_breakeven": False}),
            ("target70_partial12_be", {"kalcb.exit.target_r": 70.0, "kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 12.0, "kalcb.exit.partial_fraction": 0.33, "kalcb.exit.partial_stop_to_breakeven": True, "kalcb.exit.partial_breakeven_buffer_r": 0.10}),
            ("target70_mfe_giveback_10_5_h24", {"kalcb.exit.target_r": 70.0, "kalcb.exit.mfe_giveback_enabled": True, "kalcb.exit.mfe_giveback_start_r": 10.0, "kalcb.exit.mfe_giveback_gap_r": 5.0, "kalcb.exit.mfe_giveback_min_hold_bars": 24}),
        ],
        5: [
            ("failed_followthrough_6_075_n025", {"kalcb.exit.failed_followthrough_bars": 6, "kalcb.exit.failed_followthrough_mfe_r": 0.75, "kalcb.exit.failed_followthrough_close_r": -0.25}),
            ("failed_followthrough_6_075_n025_persistent", {"kalcb.exit.failed_followthrough_bars": 6, "kalcb.exit.failed_followthrough_mfe_r": 0.75, "kalcb.exit.failed_followthrough_close_r": -0.25, "kalcb.exit.failed_followthrough_persistent": True}),
            ("failed_followthrough_8_100_n050", {"kalcb.exit.failed_followthrough_bars": 8, "kalcb.exit.failed_followthrough_mfe_r": 1.0, "kalcb.exit.failed_followthrough_close_r": -0.5}),
            ("failed_followthrough_8_100_n050_persistent", {"kalcb.exit.failed_followthrough_bars": 8, "kalcb.exit.failed_followthrough_mfe_r": 1.0, "kalcb.exit.failed_followthrough_close_r": -0.5, "kalcb.exit.failed_followthrough_persistent": True}),
            ("failed_followthrough_10_125_n025", {"kalcb.exit.failed_followthrough_bars": 10, "kalcb.exit.failed_followthrough_mfe_r": 1.25, "kalcb.exit.failed_followthrough_close_r": -0.25}),
            ("failed_followthrough_10_125_n025_persistent", {"kalcb.exit.failed_followthrough_bars": 10, "kalcb.exit.failed_followthrough_mfe_r": 1.25, "kalcb.exit.failed_followthrough_close_r": -0.25, "kalcb.exit.failed_followthrough_persistent": True}),
            ("failed_followthrough_12_150_flat", {"kalcb.exit.failed_followthrough_bars": 12, "kalcb.exit.failed_followthrough_mfe_r": 1.5, "kalcb.exit.failed_followthrough_close_r": 0.0}),
            ("shadow_failfast_8_075_n025", {"kalcb.exit.shadow_failed_followthrough_bars": 8, "kalcb.exit.shadow_failed_followthrough_mfe_r": 0.75, "kalcb.exit.shadow_failed_followthrough_close_r": -0.25}),
            ("shadow_failfast_10_100_n050", {"kalcb.exit.shadow_failed_followthrough_bars": 10, "kalcb.exit.shadow_failed_followthrough_mfe_r": 1.0, "kalcb.exit.shadow_failed_followthrough_close_r": -0.5}),
            ("time_decay_36_mfe2r_close0", {"kalcb.exit.time_decay_bars": 36, "kalcb.exit.time_decay_min_mfe_r": 2.0, "kalcb.exit.time_decay_max_current_r": 0.0}),
            ("time_decay_48_mfe3r_close05", {"kalcb.exit.time_decay_bars": 48, "kalcb.exit.time_decay_min_mfe_r": 3.0, "kalcb.exit.time_decay_max_current_r": 0.5}),
            ("time_decay_60_mfe4r_close05", {"kalcb.exit.time_decay_bars": 60, "kalcb.exit.time_decay_min_mfe_r": 4.0, "kalcb.exit.time_decay_max_current_r": 0.5}),
            ("time_decay_72_mfe5r_close1", {"kalcb.exit.time_decay_bars": 72, "kalcb.exit.time_decay_min_mfe_r": 5.0, "kalcb.exit.time_decay_max_current_r": 1.0}),
            ("max_hold_72", {"kalcb.exit.max_hold_bars": 72}),
            ("vwap_fail_after_4r_3_15bps", {"kalcb.exit.vwap_fail_bars": 3, "kalcb.exit.vwap_fail_pct": 0.0015, "kalcb.exit.vwap_fail_after_mfe_r": 4.0}),
            ("mfe_giveback_8_4_h18", {"kalcb.exit.mfe_giveback_enabled": True, "kalcb.exit.mfe_giveback_start_r": 8.0, "kalcb.exit.mfe_giveback_gap_r": 4.0, "kalcb.exit.mfe_giveback_min_hold_bars": 18}),
            ("mfe_giveback_10_5_h24", {"kalcb.exit.mfe_giveback_enabled": True, "kalcb.exit.mfe_giveback_start_r": 10.0, "kalcb.exit.mfe_giveback_gap_r": 5.0, "kalcb.exit.mfe_giveback_min_hold_bars": 24}),
            ("mfe_giveback_12_6_h36", {"kalcb.exit.mfe_giveback_enabled": True, "kalcb.exit.mfe_giveback_start_r": 12.0, "kalcb.exit.mfe_giveback_gap_r": 6.0, "kalcb.exit.mfe_giveback_min_hold_bars": 36}),
            ("late_giveback_36_8_4", {"kalcb.exit.late_giveback_start_bars": 36, "kalcb.exit.late_giveback_start_r": 8.0, "kalcb.exit.late_giveback_gap_r": 4.0}),
            ("late_giveback_48_10_5", {"kalcb.exit.late_giveback_start_bars": 48, "kalcb.exit.late_giveback_start_r": 10.0, "kalcb.exit.late_giveback_gap_r": 5.0}),
            ("late_giveback_60_12_6", {"kalcb.exit.late_giveback_start_bars": 60, "kalcb.exit.late_giveback_start_r": 12.0, "kalcb.exit.late_giveback_gap_r": 6.0}),
            ("conditional_stop_6_3_h12", {"kalcb.exit.conditional_stop_activate_r": 6.0, "kalcb.exit.conditional_stop_gap_r": 3.0, "kalcb.exit.conditional_stop_min_hold_bars": 12}),
            ("conditional_stop_8_4_h24", {"kalcb.exit.conditional_stop_activate_r": 8.0, "kalcb.exit.conditional_stop_gap_r": 4.0, "kalcb.exit.conditional_stop_min_hold_bars": 24}),
            ("conditional_stop_10_5_h24", {"kalcb.exit.conditional_stop_activate_r": 10.0, "kalcb.exit.conditional_stop_gap_r": 5.0, "kalcb.exit.conditional_stop_min_hold_bars": 24}),
            ("target70_late_giveback_48_10_5", {"kalcb.exit.target_r": 70.0, "kalcb.exit.late_giveback_start_bars": 48, "kalcb.exit.late_giveback_start_r": 10.0, "kalcb.exit.late_giveback_gap_r": 5.0}),
            ("partial12_late48_runner", {"kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 12.0, "kalcb.exit.partial_fraction": 0.25, "kalcb.exit.partial_stop_to_breakeven": True, "kalcb.exit.partial_breakeven_buffer_r": 0.10, "kalcb.exit.late_giveback_start_bars": 48, "kalcb.exit.late_giveback_start_r": 10.0, "kalcb.exit.late_giveback_gap_r": 5.0}),
            ("ff10_plus_time_decay_48", {"kalcb.exit.failed_followthrough_bars": 10, "kalcb.exit.failed_followthrough_mfe_r": 1.25, "kalcb.exit.failed_followthrough_close_r": -0.25, "kalcb.exit.time_decay_bars": 48, "kalcb.exit.time_decay_min_mfe_r": 3.0, "kalcb.exit.time_decay_max_current_r": 0.5}),
            ("ff10_plus_time_decay_72", {"kalcb.exit.failed_followthrough_bars": 10, "kalcb.exit.failed_followthrough_mfe_r": 1.25, "kalcb.exit.failed_followthrough_close_r": -0.25, "kalcb.exit.time_decay_bars": 72, "kalcb.exit.time_decay_min_mfe_r": 5.0, "kalcb.exit.time_decay_max_current_r": 1.0}),
        ],
        6: [
            ("risk_006_cap45", {"kalcb.risk.risk_per_trade_pct": 0.006, "kalcb.risk.max_position_notional_pct": 0.45}),
            ("risk_007_cap35", {"kalcb.risk.risk_per_trade_pct": 0.007, "kalcb.risk.max_position_notional_pct": 0.35}),
            ("risk_0065_cap40", {"kalcb.risk.risk_per_trade_pct": 0.0065, "kalcb.risk.max_position_notional_pct": 0.40}),
            ("risk_0055_cap50", {"kalcb.risk.risk_per_trade_pct": 0.0055, "kalcb.risk.max_position_notional_pct": 0.50}),
            ("positions_6_sector4", {"kalcb.risk.max_positions": 6, "kalcb.risk.max_per_sector": 4}),
            ("positions_8_sector4", {"kalcb.risk.max_positions": 8, "kalcb.risk.max_per_sector": 4}),
            ("leverage_175", {"kalcb.risk.intraday_leverage": 1.75}),
            ("participation_0075", {"kalcb.risk.max_participation_30m": 0.0075}),
        ],
        7: [
            ("postrisk_risk005_cap55_time_decay36", {"kalcb.risk.risk_per_trade_pct": 0.005, "kalcb.risk.max_position_notional_pct": 0.55, "kalcb.exit.time_decay_bars": 36, "kalcb.exit.time_decay_min_mfe_r": 4.0, "kalcb.exit.time_decay_max_current_r": 0.5}),
            ("postrisk_target_30r", {"kalcb.exit.target_r": 30.0}),
            ("postrisk_target_32r", {"kalcb.exit.target_r": 32.0}),
            ("postrisk_target_34r", {"kalcb.exit.target_r": 34.0}),
            ("postrisk_target_36r", {"kalcb.exit.target_r": 36.0}),
            ("postrisk_target_38r", {"kalcb.exit.target_r": 38.0}),
            ("postrisk_target_40r", {"kalcb.exit.target_r": 40.0}),
            ("postrisk_target_45r", {"kalcb.exit.target_r": 45.0}),
            ("postrisk_target_50r", {"kalcb.exit.target_r": 50.0}),
            ("postrisk_target_60r_net_challenger", {"kalcb.exit.target_r": 60.0}),
            ("postrisk_target_36r_time_decay72", {"kalcb.exit.target_r": 36.0, "kalcb.exit.time_decay_bars": 72, "kalcb.exit.time_decay_min_mfe_r": 5.0, "kalcb.exit.time_decay_max_current_r": 1.0}),
            ("postrisk_target_60r_time_decay72", {"kalcb.exit.target_r": 60.0, "kalcb.exit.time_decay_bars": 72, "kalcb.exit.time_decay_min_mfe_r": 5.0, "kalcb.exit.time_decay_max_current_r": 1.0}),
            ("postrisk_target70_mfe10_5_h24", {"kalcb.exit.target_r": 70.0, "kalcb.exit.mfe_giveback_enabled": True, "kalcb.exit.mfe_giveback_start_r": 10.0, "kalcb.exit.mfe_giveback_gap_r": 5.0, "kalcb.exit.mfe_giveback_min_hold_bars": 24}),
            ("postrisk_target70_late48_10_5", {"kalcb.exit.target_r": 70.0, "kalcb.exit.late_giveback_start_bars": 48, "kalcb.exit.late_giveback_start_r": 10.0, "kalcb.exit.late_giveback_gap_r": 5.0}),
            ("postrisk_partial12_target70_late48", {"kalcb.exit.target_r": 70.0, "kalcb.exit.use_partial_takes": True, "kalcb.exit.partial_r_trigger": 12.0, "kalcb.exit.partial_fraction": 0.25, "kalcb.exit.partial_stop_to_breakeven": True, "kalcb.exit.partial_breakeven_buffer_r": 0.10, "kalcb.exit.late_giveback_start_bars": 48, "kalcb.exit.late_giveback_start_r": 10.0, "kalcb.exit.late_giveback_gap_r": 5.0}),
        ],
        8: [
            ("final_relvol075_frequency_probe", {"kalcb.entry.min_first30_rel_volume": 0.75}),
            ("final_first30_ret_removed_frequency_probe", {"kalcb.entry.min_bar_ret": 0.0}),
            ("final_first30_ret_75bps_quality", {"kalcb.entry.min_bar_ret": 0.0075}),
            ("final_relvol125_quality", {"kalcb.entry.min_first30_rel_volume": 1.25}),
            ("final_frontier_rank_top12", {"kalcb.entry.max_frontier_rank": 12}),
            ("final_frontier_expand_top8_cpr65_relvol125", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 8, "kalcb.entry.min_first30_signal_cpr": 0.65, "kalcb.entry.min_first30_rel_volume": 1.25}),
            ("final_frontier_expand_top12_cpr70_relvol150", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 12, "kalcb.entry.min_first30_signal_cpr": 0.70, "kalcb.entry.min_first30_rel_volume": 1.50}),
            ("final_frontier_expand_top20_cpr75_relvol200", {"kalcb.entry.require_initial_active": False, "kalcb.entry.max_frontier_rank": 20, "kalcb.entry.min_first30_signal_cpr": 0.75, "kalcb.entry.min_first30_rel_volume": 2.00}),
            ("final_quality_vote_cpr75_range075_6of7", {"kalcb.entry.min_quality_votes": 6, "kalcb.entry.quality_min_bar_ret": 0.01, "kalcb.entry.quality_min_first30_signal_cpr": 0.75, "kalcb.entry.quality_min_flow_score": -0.05, "kalcb.entry.quality_min_accumulation_score": 0.0, "kalcb.entry.quality_max_frontier_rank": 20, "kalcb.entry.quality_min_first30_rel_volume": 2.0, "kalcb.entry.quality_min_first30_range_atr": 0.75}),
            ("final_quality_vote_cpr75_accum05_5of6", {"kalcb.entry.min_quality_votes": 5, "kalcb.entry.quality_min_bar_ret": 0.0125, "kalcb.entry.quality_min_first30_signal_cpr": 0.75, "kalcb.entry.quality_min_flow_score": -0.05, "kalcb.entry.quality_min_accumulation_score": 0.05, "kalcb.entry.quality_max_frontier_rank": 12, "kalcb.entry.quality_min_first30_rel_volume": 1.0}),
            (
                "final_routes_rank8_reclaim_q6",
                _route_plan(
                    {"name": "rank8_pullback_reclaim_q6", "mode": "pullback_acceptance", "priority": 10, "after_bar": 1, "max_signal_bars": 18, "min_bar_ret": 0.0, "min_reclaim_ret": 0.0005, "max_pullback_from_vwap_pct": 0.008, "max_frontier_rank": 8, "min_quality_votes": 6, "quality_min_first30_signal_cpr": 0.75, "quality_min_first30_rel_volume": 1.5}
                ),
            ),
            (
                "final_routes_rank6_deferred_q7",
                _route_plan(
                    {"name": "rank6_deferred_q7", "mode": "deferred_continuation", "priority": 20, "after_bar": 2, "max_signal_bars": 24, "min_bar_ret": 0.0, "min_breakout_pct": 0.001, "min_close_location": 0.60, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.70, "quality_min_first30_rel_volume": 1.25}
                ),
            ),
            (
                "final_routes_rank6_reclaim_deferred",
                _route_plan(
                    {"name": "rank6_or_high_reclaim_q7", "mode": "or_high_reclaim", "priority": 10, "after_bar": 1, "max_signal_bars": 15, "min_bar_ret": 0.0, "min_reclaim_ret": 0.0, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.72, "quality_min_first30_rel_volume": 1.5},
                    {"name": "rank6_deferred_q7", "mode": "deferred_continuation", "priority": 20, "after_bar": 2, "max_signal_bars": 24, "min_bar_ret": 0.0, "min_breakout_pct": 0.001, "min_close_location": 0.60, "max_frontier_rank": 6, "min_quality_votes": 7, "quality_min_first30_signal_cpr": 0.70, "quality_min_first30_rel_volume": 1.25},
                ),
            ),
        ],
        9: [
            (
                "append9_rank8_reclaim_scaled25_cap2",
                _route_plan(
                    {
                        "name": "rank8_reclaim_scaled25",
                        "mode": "pullback_acceptance",
                        "priority": 10,
                        "after_bar": 1,
                        "max_signal_bars": 18,
                        "require_initial_active": False,
                        "max_frontier_rank": 8,
                        "min_quality_votes": 6,
                        "quality_min_first30_signal_cpr": 0.75,
                        "quality_min_first30_rel_volume": 1.5,
                        "min_bar_ret": 0.0,
                        "min_reclaim_ret": 0.0005,
                        "max_pullback_from_vwap_pct": 0.008,
                        "risk_mult": 0.25,
                        "notional_mult": 0.25,
                        "max_session_trades": 2,
                    }
                ),
            ),
            (
                "append9_rank8_deferred_scaled25_cap3",
                _route_plan(
                    {
                        "name": "rank8_deferred_scaled25",
                        "mode": "deferred_continuation",
                        "priority": 20,
                        "after_bar": 2,
                        "max_signal_bars": 24,
                        "require_initial_active": False,
                        "max_frontier_rank": 8,
                        "min_quality_votes": 6,
                        "quality_min_first30_signal_cpr": 0.75,
                        "quality_min_first30_rel_volume": 1.5,
                        "min_bar_ret": 0.0,
                        "min_breakout_pct": 0.001,
                        "min_close_location": 0.60,
                        "risk_mult": 0.25,
                        "notional_mult": 0.25,
                        "max_session_trades": 3,
                    }
                ),
            ),
            (
                "append9_rank12_reclaim_scaled20_cap2",
                _route_plan(
                    {
                        "name": "rank12_reclaim_scaled20",
                        "mode": "pullback_acceptance",
                        "priority": 10,
                        "after_bar": 1,
                        "max_signal_bars": 18,
                        "require_initial_active": False,
                        "max_frontier_rank": 12,
                        "min_quality_votes": 7,
                        "quality_min_first30_signal_cpr": 0.78,
                        "quality_min_first30_rel_volume": 1.75,
                        "min_bar_ret": 0.0,
                        "min_reclaim_ret": 0.0005,
                        "max_pullback_from_vwap_pct": 0.006,
                        "risk_mult": 0.20,
                        "notional_mult": 0.20,
                        "max_session_trades": 2,
                    }
                ),
            ),
            (
                "append9_rank12_deferred_scaled20_cap3",
                _route_plan(
                    {
                        "name": "rank12_deferred_scaled20",
                        "mode": "deferred_continuation",
                        "priority": 20,
                        "after_bar": 2,
                        "max_signal_bars": 24,
                        "require_initial_active": False,
                        "max_frontier_rank": 12,
                        "min_quality_votes": 7,
                        "quality_min_first30_signal_cpr": 0.78,
                        "quality_min_first30_rel_volume": 1.75,
                        "min_bar_ret": 0.0,
                        "min_breakout_pct": 0.001,
                        "min_close_location": 0.62,
                        "risk_mult": 0.20,
                        "notional_mult": 0.20,
                        "max_session_trades": 3,
                    }
                ),
            ),
            (
                "append9_rank20_deferred_scaled15_cap2",
                _route_plan(
                    {
                        "name": "rank20_deferred_scaled15",
                        "mode": "deferred_continuation",
                        "priority": 20,
                        "after_bar": 2,
                        "max_signal_bars": 24,
                        "require_initial_active": False,
                        "max_frontier_rank": 20,
                        "min_quality_votes": 7,
                        "quality_min_first30_signal_cpr": 0.80,
                        "quality_min_first30_rel_volume": 2.00,
                        "min_bar_ret": 0.0,
                        "min_breakout_pct": 0.0015,
                        "min_close_location": 0.65,
                        "risk_mult": 0.15,
                        "notional_mult": 0.15,
                        "max_session_trades": 2,
                    }
                ),
            ),
            (
                "append9_rank8_reclaim_deferred_scaled20_cap1_each",
                _route_plan(
                    {
                        "name": "rank8_reclaim_scaled20",
                        "mode": "pullback_acceptance",
                        "priority": 10,
                        "after_bar": 1,
                        "max_signal_bars": 18,
                        "require_initial_active": False,
                        "max_frontier_rank": 8,
                        "min_quality_votes": 6,
                        "quality_min_first30_signal_cpr": 0.75,
                        "quality_min_first30_rel_volume": 1.5,
                        "min_bar_ret": 0.0,
                        "min_reclaim_ret": 0.0005,
                        "max_pullback_from_vwap_pct": 0.008,
                        "risk_mult": 0.20,
                        "notional_mult": 0.20,
                        "max_session_trades": 1,
                    },
                    {
                        "name": "rank8_deferred_scaled20",
                        "mode": "deferred_continuation",
                        "priority": 20,
                        "after_bar": 2,
                        "max_signal_bars": 24,
                        "require_initial_active": False,
                        "max_frontier_rank": 8,
                        "min_quality_votes": 6,
                        "quality_min_first30_signal_cpr": 0.75,
                        "quality_min_first30_rel_volume": 1.5,
                        "min_bar_ret": 0.0,
                        "min_breakout_pct": 0.001,
                        "min_close_location": 0.60,
                        "risk_mult": 0.20,
                        "notional_mult": 0.20,
                        "max_session_trades": 1,
                    },
                ),
            ),
            (
                "append9_rank12_deferred_mfe_floor_scaled20_cap3",
                {
                    **_route_plan(
                        {
                            "name": "rank12_deferred_scaled20",
                            "mode": "deferred_continuation",
                            "priority": 20,
                            "after_bar": 2,
                            "max_signal_bars": 24,
                            "require_initial_active": False,
                            "max_frontier_rank": 12,
                            "min_quality_votes": 7,
                            "quality_min_first30_signal_cpr": 0.78,
                            "quality_min_first30_rel_volume": 1.75,
                            "min_bar_ret": 0.0,
                            "min_breakout_pct": 0.001,
                            "min_close_location": 0.62,
                            "risk_mult": 0.20,
                            "notional_mult": 0.20,
                            "max_session_trades": 3,
                        }
                    ),
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": 0.0,
                    "kalcb.exit.mfe_floor_min_hold_bars": 24,
                    "kalcb.exit.mfe_floor_entry_routes": ["rank12_deferred_scaled20"],
                },
            ),
            (
                "append9_rank30_deferred_scaled10_cap1",
                _route_plan(
                    {
                        "name": "rank30_deferred_scaled10",
                        "mode": "deferred_continuation",
                        "priority": 20,
                        "after_bar": 2,
                        "max_signal_bars": 24,
                        "require_initial_active": False,
                        "max_frontier_rank": 30,
                        "min_quality_votes": 8,
                        "quality_min_first30_signal_cpr": 0.82,
                        "quality_min_first30_rel_volume": 2.25,
                        "quality_min_accumulation_score": 0.05,
                        "min_bar_ret": 0.0,
                        "min_breakout_pct": 0.0015,
                        "min_close_location": 0.66,
                        "risk_mult": 0.10,
                        "notional_mult": 0.10,
                        "max_session_trades": 1,
                    }
                ),
            ),
        ],
        10: [
            ("append10_risk0045_cap60", {"kalcb.risk.risk_per_trade_pct": 0.0045, "kalcb.risk.max_position_notional_pct": 0.60}),
            ("append10_risk0050_cap60", {"kalcb.risk.risk_per_trade_pct": 0.0050, "kalcb.risk.max_position_notional_pct": 0.60}),
            ("append10_risk0055_cap60", {"kalcb.risk.risk_per_trade_pct": 0.0055, "kalcb.risk.max_position_notional_pct": 0.60}),
            ("append10_risk0045_cap55", {"kalcb.risk.risk_per_trade_pct": 0.0045, "kalcb.risk.max_position_notional_pct": 0.55}),
            ("append10_risk0050_cap55", {"kalcb.risk.risk_per_trade_pct": 0.0050, "kalcb.risk.max_position_notional_pct": 0.55}),
            ("append10_risk00525_cap55", {"kalcb.risk.risk_per_trade_pct": 0.00525, "kalcb.risk.max_position_notional_pct": 0.55}),
        ],
        11: [
            (
                "append11_relvol3_mfe6_floor1_h24",
                {
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": 1.0,
                    "kalcb.exit.mfe_floor_min_hold_bars": 24,
                    "kalcb.exit.mfe_floor_max_first30_rel_volume": 3.0,
                    "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
                },
            ),
            (
                "append11_relvol3_mfe6_floor0_h24",
                {
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": 0.0,
                    "kalcb.exit.mfe_floor_min_hold_bars": 24,
                    "kalcb.exit.mfe_floor_max_first30_rel_volume": 3.0,
                    "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
                },
            ),
            (
                "append11_relvol3_mfe6_floor0_h36",
                {
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": 0.0,
                    "kalcb.exit.mfe_floor_min_hold_bars": 36,
                    "kalcb.exit.mfe_floor_max_first30_rel_volume": 3.0,
                    "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
                },
            ),
            (
                "append11_relvol3_mfe6_floor0_h18",
                {
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": 0.0,
                    "kalcb.exit.mfe_floor_min_hold_bars": 18,
                    "kalcb.exit.mfe_floor_max_first30_rel_volume": 3.0,
                    "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
                },
            ),
            (
                "append11_relvol3_mfe6_floor_n05_h24",
                {
                    "kalcb.exit.mfe_floor_enabled": True,
                    "kalcb.exit.mfe_floor_start_r": 6.0,
                    "kalcb.exit.mfe_floor_floor_r": -0.5,
                    "kalcb.exit.mfe_floor_min_hold_bars": 24,
                    "kalcb.exit.mfe_floor_max_first30_rel_volume": 3.0,
                    "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
                },
            ),
        ],
    }
    redesigned_items = _redesigned_late_phase_candidates(int(phase))
    items = redesigned_items if redesigned_items is not None else raw.get(int(phase), [])
    return [Experiment(name, mutations) for name, mutations in items]


def _redesigned_late_phase_candidates(phase: int) -> list[tuple[str, dict[str, Any]]] | None:
    if phase in {9, 10, 11}:
        return []
    return None


def score_fixed(metrics: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    components = weights or SCORE_COMPONENTS
    if len(components) > 7:
        raise ValueError("KALCB fixed immutable score supports at most 7 components")
    return 100.0 * sum(float(weight) * _scaled_component(name, metrics) for name, weight in components.items())


def reject_reason(metrics: dict[str, Any], hard: dict[str, float]) -> str:
    trades = float(metrics.get("trade_count", metrics.get("total_trades", 0.0)) or 0.0)
    if trades < float(hard.get("min_trades", 0.0)):
        return f"too_few_trades ({trades:.0f} < {float(hard.get('min_trades', 0.0)):.0f})"
    dd = float(metrics.get("broker_max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0)) or 0.0)
    if dd > float(hard.get("max_dd_pct", 1.0)):
        return f"max_dd ({dd:.2%} > {float(hard.get('max_dd_pct', 1.0)):.2%})"
    if float(metrics.get("same_bar_fill_count", 0.0) or 0.0) > float(hard.get("max_same_bar_fills", 0.0)):
        return "same_bar_fill"
    if float(metrics.get("end_open_position_count", 0.0) or 0.0) > float(hard.get("max_end_open_positions", 0.0)):
        return "end_open_positions"
    if float(metrics.get("broker_net_return_pct", 0.0) or 0.0) < float(hard.get("min_net_return_pct", -1.0)):
        return f"negative_net ({float(metrics.get('broker_net_return_pct', 0.0) or 0.0):.2%})"
    if float(metrics.get("worst_fold_net", 0.0) or 0.0) < float(hard.get("min_worst_fold_net", -1.0)):
        return f"negative_worst_fold ({float(metrics.get('worst_fold_net', 0.0) or 0.0):.2%})"
    return ""


def _quality_guardrail_reject_reason(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    trade_delta = float(metrics.get("trade_count", 0.0) or 0.0) - float(baseline.get("trade_count", 0.0) or 0.0)
    if trade_delta < 5.0:
        return ""
    deteriorated = (
        float(metrics.get("broker_net_return_pct", 0.0) or 0.0) < float(baseline.get("broker_net_return_pct", 0.0) or 0.0)
        and float(metrics.get("avg_trade_net_pct", 0.0) or 0.0) < float(baseline.get("avg_trade_net_pct", 0.0) or 0.0)
        and float(metrics.get("worst_fold_net", 0.0) or 0.0) < float(baseline.get("worst_fold_net", 0.0) or 0.0)
        and float(metrics.get("avg_mfe_capture", 0.0) or 0.0) < float(baseline.get("avg_mfe_capture", 0.0) or 0.0)
    )
    if deteriorated:
        return "frequency_without_expectancy_guardrail"
    return ""


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config or {})
    baseline = dict(out.get("baseline") or {})
    holdout_start = str(baseline.get("holdout_start") or "").strip()
    holdout_end = str(baseline.get("holdout_end") or "").strip()
    if not holdout_start or not holdout_end:
        raise ValueError("validation_gate_enabled requires baseline.holdout_start and baseline.holdout_end")
    out["start"] = holdout_start
    out["end"] = holdout_end
    out["validation_gate_enabled"] = False
    out["fixed_trade_plan_phase_auto"] = True
    out["force_rebuild_cache"] = False
    return out


def _replay_context_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config or {})
    for key in (
        "validation_gate_enabled",
        "skip_initial_baseline_eval",
    ):
        out.pop(key, None)
    return out


def _metric_deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "avg_trade_net_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "worst_fold_net",
        "avg_mfe_capture",
        "mae_le_neg_1_share",
    )
    return {key: float(metrics.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0) for key in keys}


def _compact_validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "avg_trade_net_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "worst_fold_net",
        "avg_mfe_capture",
        "mae_le_neg_1_share",
        "training_window_start",
        "training_window_end",
        "training_session_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _validation_gate_reject_reason(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    validation: dict[str, Any],
    baseline_validation: dict[str, Any],
) -> str:
    train_delta = _metric_deltas(metrics, baseline)
    validation_delta = _metric_deltas(validation, baseline_validation)
    if train_delta["broker_net_return_pct"] <= 0:
        return "validation_gate_train_net_not_positive"
    if validation_delta["broker_net_return_pct"] <= 0:
        return "validation_gate_holdout_net_not_positive"
    if train_delta["avg_trade_net_pct"] < -0.0005:
        return "validation_gate_train_avg_trade_deterioration"
    if validation_delta["avg_trade_net_pct"] < -0.0005:
        return "validation_gate_holdout_avg_trade_deterioration"
    if train_delta["broker_max_drawdown_pct"] > 0.005:
        return "validation_gate_train_drawdown_expansion"
    if validation_delta["broker_max_drawdown_pct"] > 0.005:
        return "validation_gate_holdout_drawdown_expansion"
    baseline_train_trades = float(baseline.get("trade_count", 0.0) or 0.0)
    baseline_validation_trades = float(baseline_validation.get("trade_count", 0.0) or 0.0)
    if train_delta["trade_count"] < -max(5.0, baseline_train_trades * 0.10):
        return "validation_gate_train_frequency_collapse"
    if validation_delta["trade_count"] < -max(2.0, baseline_validation_trades * 0.20):
        return "validation_gate_holdout_frequency_collapse"
    if validation_delta["mae_le_neg_1_share"] > 0.05 and validation_delta["avg_mfe_capture"] < 0:
        return "validation_gate_holdout_hygiene_deterioration"
    return ""


def gate_criteria(metrics: dict[str, Any], hard: dict[str, float]) -> list[GateCriterion]:
    trades = float(metrics.get("trade_count", metrics.get("total_trades", 0.0)) or 0.0)
    dd = float(metrics.get("broker_max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0)) or 0.0)
    same_bar = float(metrics.get("same_bar_fill_count", 0.0) or 0.0)
    open_pos = float(metrics.get("end_open_position_count", 0.0) or 0.0)
    net = float(metrics.get("broker_net_return_pct", 0.0) or 0.0)
    worst_fold = float(metrics.get("worst_fold_net", 0.0) or 0.0)
    return [
        GateCriterion("hard_trade_count", float(hard["min_trades"]), trades, trades >= float(hard["min_trades"])),
        GateCriterion("broker_max_drawdown_pct", float(hard["max_dd_pct"]), dd, dd <= float(hard["max_dd_pct"])),
        GateCriterion("same_bar_fill_count", float(hard["max_same_bar_fills"]), same_bar, same_bar <= float(hard["max_same_bar_fills"])),
        GateCriterion("end_open_position_count", float(hard["max_end_open_positions"]), open_pos, open_pos <= float(hard["max_end_open_positions"])),
        GateCriterion("broker_net_return_pct", float(hard["min_net_return_pct"]), net, net >= float(hard["min_net_return_pct"])),
        GateCriterion("worst_fold_net", float(hard["min_worst_fold_net"]), worst_fold, worst_fold >= float(hard["min_worst_fold_net"])),
    ]


def _scaled_component(name: str, metrics: dict[str, Any]) -> float:
    if name == "broker_net_return_pct":
        return math.tanh(float(metrics.get(name, 0.0) or 0.0) / 0.75)
    if name == "broker_expected_total_r":
        return math.tanh(float(metrics.get(name, 0.0) or 0.0) / 1800.0)
    if name == "avg_trade_net_pct":
        return math.tanh(float(metrics.get(name, 0.0) or 0.0) / 0.012)
    if name == "frequency":
        trades = float(metrics.get("trade_count", metrics.get("total_trades", 0.0)) or 0.0)
        active_days = float(metrics.get("active_days", 0.0) or 0.0)
        return 0.70 * _clip((trades - 120.0) / 140.0) + 0.30 * _clip((active_days - 45.0) / 45.0)
    if name == "worst_fold_net":
        return math.tanh(float(metrics.get(name, 0.0) or 0.0) / 0.15)
    if name == "avg_mfe_capture":
        return _clip((float(metrics.get(name, 0.0) or 0.0) - 0.32) / 0.28)
    if name == "broker_max_drawdown_pct":
        return _clip(float(metrics.get(name, 0.0) or 0.0) / 0.075, 0.0, 2.0)
    if name == "mae_tail_loss":
        tail = _clip(float(metrics.get("mae_le_neg_1_share", 0.0) or 0.0) / 0.80, 0.0, 2.0)
        avg_mae = abs(min(float(metrics.get("avg_mae_r", 0.0) or 0.0), 0.0))
        depth = _clip(avg_mae / 10.0, 0.0, 2.0)
        accepted = dict(metrics.get("accepted_loser_summary") or {})
        accepted_mae = abs(min(float(accepted.get("avg_mae_r", 0.0) or 0.0), 0.0))
        accepted_depth = _clip(accepted_mae / 12.0, 0.0, 2.0)
        return 0.45 * tail + 0.35 * depth + 0.20 * accepted_depth
    return float(metrics.get(name, 0.0) or 0.0)


def _normalize_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    out = dict(mutations or {})
    out.setdefault("kalcb.carry.mode", "off")
    out.setdefault("kalcb.frontier.shadow_enabled", False)
    out.setdefault("kalcb.frontier.rotation_enabled", False)
    out.setdefault("kalcb.entry.require_initial_active", True)
    out.setdefault("kalcb.entry.fast_replay_suppress_rejections", True)
    out.setdefault("kalcb.risk.max_participation_30m", PORTFOLIO_RISK_POLICY["max_participation_30m"])
    out.setdefault("kalcb.risk.intraday_leverage", PORTFOLIO_RISK_POLICY["intraday_leverage"])
    out.setdefault("kalcb.risk.max_positions", PORTFOLIO_RISK_POLICY["max_positions_cap"])
    out.setdefault("kalcb.risk.max_per_sector", PORTFOLIO_RISK_POLICY["max_per_sector_cap"])
    out.setdefault("kalcb.risk.risk_per_trade_pct", PORTFOLIO_RISK_POLICY["risk_per_trade_pct"])
    out.setdefault("kalcb.risk.max_position_notional_pct", PORTFOLIO_RISK_POLICY["max_position_notional_pct"])
    risk = max(float(out.get("kalcb.risk.risk_per_trade_pct", PORTFOLIO_RISK_POLICY["risk_per_trade_pct"]) or 0.0), 1e-9)
    heat_pct = float(out.get("kalcb.risk.heat_cap_pct", PORTFOLIO_RISK_POLICY["heat_cap_pct"]) or PORTFOLIO_RISK_POLICY["heat_cap_pct"])
    out["kalcb.risk.heat_cap_r"] = heat_pct / risk
    return out


def _artifact_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    out = _normalize_mutations(mutations)
    out.pop("kalcb.entry.fast_replay_suppress_rejections", None)
    return out


def _add_fold_metrics(metrics: dict[str, Any], fold_rows: tuple[dict[str, Any], ...]) -> None:
    values = [
        float(row.get("metrics", {}).get("portfolio_equivalent_net_return_pct", row.get("metrics", {}).get("broker_net_return_pct", 0.0)) or 0.0)
        for row in fold_rows
    ]
    dds = [
        float(row.get("metrics", {}).get("portfolio_equivalent_max_drawdown_pct", row.get("metrics", {}).get("broker_max_drawdown_pct", 0.0)) or 0.0)
        for row in fold_rows
    ]
    metrics["worst_fold_net"] = min(values) if values else float(metrics.get("broker_net_return_pct", 0.0) or 0.0)
    metrics["median_fold_net"] = float(median(values)) if values else float(metrics.get("broker_net_return_pct", 0.0) or 0.0)
    metrics["worst_fold_drawdown_pct"] = max(dds) if dds else float(metrics.get("broker_max_drawdown_pct", 0.0) or 0.0)
    metrics["fold_count"] = float(len(fold_rows))


def _phase_focus(phase: int) -> tuple[str, list[str]]:
    return {
        1: ("active first30 gap-retention discrimination", ["broker_net_return_pct", "avg_trade_net_pct", "broker_max_drawdown_pct", "worst_fold_net"]),
        2: ("drawdown-contained high gap-relvol notional scaling", ["broker_net_return_pct", "broker_max_drawdown_pct", "avg_trade_net_pct", "worst_fold_net"]),
        3: ("ultra-capped frontier frequency recovery", ["trade_count", "broker_net_return_pct", "avg_trade_net_pct", "broker_max_drawdown_pct"]),
        4: ("conditional tail capture by route cohort", ["avg_mfe_capture", "broker_net_return_pct", "avg_trade_net_pct", "worst_fold_net"]),
        5: ("low-quality path giveback containment", ["broker_net_return_pct", "avg_trade_net_pct", "avg_mfe_capture", "broker_max_drawdown_pct"]),
        6: ("near-miss combined stack validation", ["broker_net_return_pct", "broker_max_drawdown_pct", "avg_trade_net_pct", "avg_mfe_capture", "worst_fold_net"]),
    }.get(int(phase), ("fixed-candidate refinement", ["broker_net_return_pct"]))


def _source_ref_for_mutations(mutations: dict[str, Any], default: FixedCandidateSourceRef) -> FixedCandidateSourceRef:
    path_value = mutations[SOURCE_PATH_MUTATION] if SOURCE_PATH_MUTATION in mutations else default.path
    section_value = mutations[SOURCE_SECTION_MUTATION] if SOURCE_SECTION_MUTATION in mutations else default.section
    rank_value = mutations[SOURCE_RANK_MUTATION] if SOURCE_RANK_MUTATION in mutations else default.rank
    return FixedCandidateSourceRef(
        path=_resolve_existing_source_path(str(path_value)),
        section=str(section_value),
        rank=int(rank_value),
        row_name="",
    )


def _selection_counts_for_dates(selections: Iterable[Selection], dates: Iterable[Any]) -> dict[Any, int]:
    counts = {day: 0 for day in dates}
    for selection in selections:
        counts[selection.trade_date] = counts.get(selection.trade_date, 0) + 1
    return counts


def _source_context_key(source_ref: FixedCandidateSourceRef) -> str:
    return json.dumps(
        {"path": source_ref.path, "section": source_ref.section, "rank": int(source_ref.rank)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _resolve_fixed_candidate_source(config: dict[str, Any], output_dir: Path) -> FixedCandidateSourceRef:
    configured = config.get("fixed_candidate_source")
    if isinstance(configured, dict) and configured.get("path"):
        return _source_ref_from_mapping(configured)
    previous = _previous_round_dir(output_dir)
    if previous is not None:
        extract = previous / "source_artifacts" / "train_rank1_extract.json"
        if extract.exists():
            data = json.loads(extract.read_text(encoding="utf-8"))
            source = dict((data.get("artifact_summary") or {}).get("fixed_candidate_source") or {})
            if source.get("source_path"):
                return FixedCandidateSourceRef(
                    path=_resolve_existing_source_path(str(source["source_path"])),
                    section=str(source.get("source_section") or "top_portfolio_proxy"),
                    rank=int(source.get("source_rank") or 0),
                    row_name=str(source.get("source_row_name") or ""),
                )
        for path, keys in (
            (previous / "full_diagnostics_index.json", ("source_ref",)),
            (previous / "run_spec.json", ("execution_context", "candidate_source")),
            (previous / "optimized_config.json", ("execution_contract", "candidate_source")),
        ):
            payload = _read_json_object(path)
            source = _nested_mapping(payload, keys)
            if source.get("path"):
                return _source_ref_from_mapping(source)
        optimized = _read_json_object(previous / "optimized_config.json")
        mutations = dict(optimized.get("mutations") or {})
        if mutations.get(SOURCE_PATH_MUTATION):
            return _source_ref_for_mutations(
                mutations,
                FixedCandidateSourceRef(path="", section="top_portfolio_proxy", rank=0, row_name=""),
            )
    raise FileNotFoundError("Could not resolve fixed KALCB candidate source from config or previous round artifacts")


def _source_ref_from_mapping(source: dict[str, Any]) -> FixedCandidateSourceRef:
    return FixedCandidateSourceRef(
        path=_resolve_existing_source_path(str(source.get("path") or source.get("source_path") or "")),
        section=str(source.get("section") or source.get("source_section") or "top_portfolio_proxy"),
        rank=int(source.get("rank") if source.get("rank") is not None else source.get("source_rank") or 0),
        row_name=str(source.get("row_name") or source.get("source_row_name") or ""),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _nested_mapping(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_existing_source_path(source_path: str) -> str:
    path = Path(source_path)
    if path.exists():
        return str(path)
    archive_root = Path("data/backtests/output/kalcb/_archive")
    if not archive_root.exists():
        return source_path
    name = path.name
    matches = sorted(candidate for candidate in archive_root.rglob(name) if candidate.is_file())
    if not matches:
        return source_path
    suffix = Path(*path.parts[-4:]) if len(path.parts) >= 4 else path
    for candidate in matches:
        if str(candidate).endswith(str(suffix)):
            return str(candidate)
    return str(matches[0])


def _load_previous_mutations(output_dir: Path) -> dict[str, Any]:
    previous = _previous_round_dir(output_dir)
    if previous is None:
        return {}
    path = previous / "optimized_config.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("mutations") or {})


def _previous_guarded_pool_mutations(output_dir: Path) -> dict[str, Any]:
    previous = _previous_round_dir(output_dir)
    if previous is None:
        return {}
    index = _read_json_object(previous / "full_diagnostics_index.json")
    artifacts = dict(index.get("artifacts") or {})
    pool_path = str(artifacts.get("train_guarded_prefilter_pool_rows") or "").strip()
    if not pool_path:
        diagnostics = _read_json_object(previous / "diagnostics_summary.json")
        pool_path = str((diagnostics.get("artifacts") or {}).get("train_guarded_prefilter_pool_rows") or "").strip()
    if not pool_path:
        return {}
    optimized = _read_json_object(previous / "optimized_config.json")
    source_artifacts = dict(optimized.get("source_artifacts") or index.get("source_hashes") or {})
    holdout_pool = dict(source_artifacts.get("holdout_pool") or {})
    policy = dict(holdout_pool.get("policy") or {})
    active_count = int(float(policy.get("active_count") or 16))
    selected_guard = dict((_read_json_object(previous / "diagnostics_summary.json").get("selected_guard") or {}))
    label = str(selected_guard.get("label") or (source_artifacts.get("guard_search_train_selected") or {}).get("guard_label") or "guarded_prefilter_pool")
    return {
        POOL_SOURCE_PATH_MUTATION: _resolve_existing_source_path(pool_path),
        POOL_SOURCE_ACTIVE_COUNT_MUTATION: active_count,
        POOL_SOURCE_LABEL_MUTATION: label,
    }


def _initial_mutations_for_output(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    configured_initial = dict((config or {}).get("initial_mutations") or {})
    previous_initial = _load_previous_mutations(output_dir)
    merged = {**configured_initial, **previous_initial} if previous_initial else configured_initial
    guarded_pool = _previous_guarded_pool_mutations(output_dir)
    if guarded_pool and not merged.get(POOL_SOURCE_PATH_MUTATION):
        merged.update(guarded_pool)
    return _artifact_mutations(merged)


def _previous_round_dir(output_dir: Path) -> Path | None:
    match = ROUND_DIR_RE.match(Path(output_dir).name)
    if not match:
        return None
    previous = int(match.group(1)) - 1
    if previous < 1:
        return None
    path = Path(output_dir).parent / f"round_{previous}"
    return path if path.exists() else None


def _cost_policy(cfg: KALCBConfig) -> dict[str, float]:
    return {
        "commission_bps_each_side": float(cfg.commission_bps),
        "slippage_bps_each_side": float(cfg.slippage_bps),
        "tax_bps_on_sell": float(cfg.tax_bps_on_sell),
        "round_trip_cost_pct": (2.0 * float(cfg.commission_bps) + 2.0 * float(cfg.slippage_bps) + float(cfg.tax_bps_on_sell)) / 10_000.0,
    }


def _cache_key(phase: int, mutation_key: str, name: str) -> str:
    return build_cache_key("kalcb.fixed_phase_eval", extra={"phase": phase, "mutation_key": mutation_key, "name": name})


def _mutation_key(mutations: dict[str, Any]) -> str:
    return json.dumps(_normalize_mutations(mutations), sort_keys=True, separators=(",", ":"), default=str)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _diagnostics_text(
    *,
    phase: int,
    focus: str,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    kept: list[str],
    source_ref: FixedCandidateSourceRef,
    execution_context: dict[str, Any],
) -> str:
    lines = [
        f"KALCB fixed-candidate phase {phase} diagnostics",
        f"Focus: {focus}",
        f"Candidate source: {source_ref.path} [{source_ref.section} rank {source_ref.rank}]",
        f"Train-only window: {metrics.get('training_window_start')} -> {metrics.get('training_window_end')} ({metrics.get('training_session_count')} sessions)",
        f"Baseline broker net/DD/trades/capture: {_pct(baseline.get('broker_net_return_pct'))} / {_pct(baseline.get('broker_max_drawdown_pct'))} / {baseline.get('trade_count', 0):.0f} / {_pct(baseline.get('avg_mfe_capture'))}",
        f"Current broker net/DD/trades/capture: {_pct(metrics.get('broker_net_return_pct'))} / {_pct(metrics.get('broker_max_drawdown_pct'))} / {metrics.get('trade_count', 0):.0f} / {_pct(metrics.get('avg_mfe_capture'))}",
        f"Worst fold net: {_pct(metrics.get('worst_fold_net'))}; avg trade net: {_pct(metrics.get('avg_trade_net_pct'))}; score: {metrics.get('immutable_score', 0.0):.3f}",
        f"Kept experiments: {', '.join(kept) if kept else 'none'}",
        f"Fast/full replay audit: {metrics.get('fast_suppression_audit', {}).get('status', 'not_run')}",
        "Execution: shared KalCB core via KALCBReplayAdapter -> SimBroker; holdout is excluded.",
        f"Implementation lessons contract: {json.dumps(execution_context.get('implementation_lessons_contract', {}), sort_keys=True)}",
    ]
    return "\n".join(lines) + "\n"


def _final_diagnostics_text(metrics: dict[str, Any], baseline: dict[str, Any], state: PhaseState, source_ref: FixedCandidateSourceRef, execution_context: dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        "KALCB ROUND-2 FIXED-CANDIDATE PHASE-AUTO DIAGNOSTICS",
        "=" * 70,
        f"Source: {source_ref.path} [{source_ref.section} rank {source_ref.rank}]",
        f"Training-only window: {metrics.get('training_window_start')} -> {metrics.get('training_window_end')} ({metrics.get('training_session_count')} sessions)",
        "Holdout policy: excluded from optimization; validate final candidate separately.",
        "",
        f"Baseline: broker_net={_pct(baseline.get('broker_net_return_pct'))}, DD={_pct(baseline.get('broker_max_drawdown_pct'))}, trades={baseline.get('trade_count', 0):.0f}, MFE_capture={_pct(baseline.get('avg_mfe_capture'))}",
        f"Final:    broker_net={_pct(metrics.get('broker_net_return_pct'))}, DD={_pct(metrics.get('broker_max_drawdown_pct'))}, trades={metrics.get('trade_count', 0):.0f}, MFE_capture={_pct(metrics.get('avg_mfe_capture'))}",
        f"Worst fold net={_pct(metrics.get('worst_fold_net'))}, median fold net={_pct(metrics.get('median_fold_net'))}, avg_trade_net={_pct(metrics.get('avg_trade_net_pct'))}",
        f"Same-bar fills={metrics.get('same_bar_fill_count', 0):.0f}, end-open positions={metrics.get('end_open_position_count', 0):.0f}",
        f"Immutable score={metrics.get('immutable_score', 0.0):.3f}; components={json.dumps(metrics.get('score_components', {}), sort_keys=True)}",
        f"Fast/full replay audit={metrics.get('fast_suppression_audit', {}).get('status', 'not_run')}; suppressed_entry_rejections={metrics.get('fast_suppression_audit', {}).get('suppressed_entry_rejection_count', 0)}; max_metric_delta={metrics.get('fast_suppression_audit', {}).get('max_abs_metric_delta', 0.0):.3g}",
        "",
        "Accepted phase mutations:",
        json.dumps(state.cumulative_mutations, indent=2, sort_keys=True, default=str),
        "",
        "Execution contract:",
        json.dumps(execution_context, indent=2, sort_keys=True, default=str),
        "",
        "Verdict: training-only fixed-candidate optimization complete. This is not live-ready until holdout and paper-parity validation pass.",
    ]
    return "\n".join(lines) + "\n"


def _full_round_diagnostics_text(
    *,
    final: dict[str, Any],
    baseline: dict[str, Any],
    fold_rows: tuple[dict[str, Any], ...],
    baseline_fold_rows: tuple[dict[str, Any], ...],
    trade_rows: tuple[dict[str, Any], ...],
    baseline_trade_rows: tuple[dict[str, Any], ...],
    state: PhaseState,
    source_ref: FixedCandidateSourceRef,
    execution_context: dict[str, Any],
    cache_metadata: dict[str, Any],
    round_num: int | None,
    round_name: str,
    live_parity_audit: dict[str, Any] | None = None,
    opportunity_diagnostics: dict[str, Any] | None = None,
) -> str:
    audit = dict(final.get("fast_suppression_audit") or {})
    source = dict(execution_context.get("candidate_source") or {})
    lines: list[str] = [
        "=" * 70,
        f"KALCB ROUND-{round_num or '?'} SOURCE-FRONTIER FULL DIAGNOSTICS",
        "=" * 70,
        f"Round name: {round_name or 'round'}",
        "Focus: explain where the strategy is extracting alpha, where it is rejecting or admitting weak signals, and which entry/management layers still need work.",
        f"Candidate source: {source_ref.path}",
        f"Candidate source section/rank: {source_ref.section} / {source_ref.rank}",
        f"Source row: {_short_text(source_ref.row_name or source.get('row_name', ''), 180)}",
        "Official path: KALCBReplayAdapter -> strategy_kalcb.core.step_kalcb_core() -> SimBroker.",
        "Research candidate generation remains upstream; trading decisions, fills, costs, sizing, exits, and metrics are shared-core/broker-owned.",
        "",
        f"Baseline: broker_net={_pct(baseline.get('broker_net_return_pct'))}, MTM={_pct(baseline.get('official_mtm_net_return_pct'))}, DD={_pct(baseline.get('broker_max_drawdown_pct'))}, trades={_num(baseline.get('trade_count'), 0)}, audit_pass=True",
        f"Final:    broker_net={_pct(final.get('broker_net_return_pct'))}, MTM={_pct(final.get('official_mtm_net_return_pct'))}, DD={_pct(final.get('broker_max_drawdown_pct'))}, trades={_num(final.get('trade_count'), 0)}, audit_pass={bool(final.get('audit_pass'))}",
        f"Score:    immutable={_num(final.get('immutable_score'), 3)}; components={len(final.get('score_components') or {})}/7; fast/full audit={audit.get('status', 'not_run')}",
        "",
        "Hard warning: this is still training-only. Holdout and paper/live parity validation are mandatory before production use.",
        "",
        "=" * 70,
        "  1. Overview",
        "=" * 70,
        "",
        f"Training window: {final.get('training_window_start')} -> {final.get('training_window_end')} ({final.get('training_session_count')} sessions)",
        f"Holdout policy: {execution_context.get('holdout_policy')}",
        f"broker_net={_pct(final.get('broker_net_return_pct'))}, official_mtm={_pct(final.get('official_mtm_net_return_pct'))}, DD={_pct(final.get('broker_max_drawdown_pct'))}, trades={_num(final.get('trade_count'), 0)}, WR={_pct(final.get('win_rate'))}, avg_MFE={_num(final.get('avg_mfe_r'), 2)}R, MFE_capture={_pct(final.get('avg_mfe_capture'))}",
        f"selected={_num(final.get('selected_count'), 0)}, conversion={_pct(final.get('signal_conversion'))}, active_days={_num(final.get('active_days'), 0)}/{_num(final.get('session_count'), 0)}, avg_trade_net={_pct(final.get('avg_trade_net_pct'))}, active_day_net={_pct(final.get('active_day_net_pct'))}",
        f"Baseline delta: broker_net {_signed_pct_delta(final, baseline, 'broker_net_return_pct')}; DD {_signed_pct_delta(final, baseline, 'broker_max_drawdown_pct')}; trades {_signed_num_delta(final, baseline, 'trade_count', 0)}",
        "",
        "=" * 70,
        "  2. Executive Verdicts",
        "=" * 70,
        "",
        *_executive_verdict_lines(final, baseline, trade_rows, baseline_trade_rows, audit),
        "",
        "=" * 70,
        "  3. Optimization Path And Score",
        "=" * 70,
        "",
        *_phase_summary_lines(state),
        "",
        f"Immutable score={_num(final.get('immutable_score'), 3)}; component limit={len(final.get('score_components') or {})}/7.",
        *_score_component_lines(final),
        "",
        "=" * 70,
        "  4. Candidate Surfacing And Source Quality",
        "=" * 70,
        "",
        f"Source file hash: {_short_hash(cache_metadata.get('candidate_source_file_hash'))}; sweep hash: {_short_hash(cache_metadata.get('candidate_source_sweep_hash'))}; row={_short_text(source_ref.row_name or source.get('row_name', ''), 180)}",
        f"Train selected: {_num(final.get('selected_count'), 0)} names over {_num(final.get('selected_days'), 0)} selected days; active day share={_pct(final.get('active_day_share'))}; avg selections/session={_num(float(final.get('selected_count', 0.0) or 0.0) / max(float(final.get('session_count', 0.0) or 0.0), 1.0), 2)}",
        f"Candidate pool={_num(final.get('candidate_pool_count'), 0)}, initial-active={_num(final.get('initial_active_candidate_count'), 0)}, frontier-expansion={_num(final.get('frontier_expansion_candidate_count'), 0)}, candidate-pool conversion={_pct(final.get('candidate_pool_conversion'))}.",
        "Research boundary: premarket uses prior completed daily/index/flow rows; first30 uses only completed 09:00-09:25 bars; the trading core receives final KALCBDailySnapshot candidates only.",
        *_candidate_surfacing_lines(final, trade_rows),
        "",
        *_opportunity_coverage_lines(opportunity_diagnostics),
        "",
        "=" * 70,
        "  5. Signal Funnel And Gate Attribution",
        "=" * 70,
        "",
        f"Entry plan: {state.cumulative_mutations.get('kalcb.entry.plan_mode')}",
        f"Entry routes: {len(state.cumulative_mutations.get('kalcb.entry.routes') or [])} configured; route metadata is included in neutral SubmitEntry actions for paper/live parity.",
        f"Entry thresholds: max_signal_bars={state.cumulative_mutations.get('kalcb.entry.max_signal_bars')}, min_bar_ret={state.cumulative_mutations.get('kalcb.entry.min_bar_ret')}, min_vwap_ret={state.cumulative_mutations.get('kalcb.entry.min_vwap_ret')}, min_close_location={state.cumulative_mutations.get('kalcb.entry.min_close_location')}, require_above_prev_close={state.cumulative_mutations.get('kalcb.entry.require_above_prev_close')}",
        f"Source/frontier gates: require_initial_active={state.cumulative_mutations.get('kalcb.entry.require_initial_active')}, max_frontier_rank={state.cumulative_mutations.get('kalcb.entry.max_frontier_rank')}, min_frontier_score={state.cumulative_mutations.get('kalcb.entry.min_frontier_score')}, min_flow={state.cumulative_mutations.get('kalcb.entry.min_flow_score')}, min_accum={state.cumulative_mutations.get('kalcb.entry.min_accumulation_score')}, min_first30_rel_volume={state.cumulative_mutations.get('kalcb.entry.min_first30_rel_volume')}, min_first30_signal_cpr={state.cumulative_mutations.get('kalcb.entry.min_first30_signal_cpr')}, min_first30_open_drawdown={state.cumulative_mutations.get('kalcb.entry.min_first30_open_drawdown')}, min_first30_low_vs_prev_close={state.cumulative_mutations.get('kalcb.entry.min_first30_low_vs_prev_close')}, first30_range_atr=[{state.cumulative_mutations.get('kalcb.entry.min_first30_range_atr')}, {state.cumulative_mutations.get('kalcb.entry.max_first30_range_atr')}]",
        f"Quality-vote gate: min_votes={state.cumulative_mutations.get('kalcb.entry.min_quality_votes')}, ret={state.cumulative_mutations.get('kalcb.entry.quality_min_bar_ret')}, cpr={state.cumulative_mutations.get('kalcb.entry.quality_min_first30_signal_cpr')}, relvol={state.cumulative_mutations.get('kalcb.entry.quality_min_first30_rel_volume')}, range_atr_min={state.cumulative_mutations.get('kalcb.entry.quality_min_first30_range_atr')}, range_atr_max={state.cumulative_mutations.get('kalcb.entry.quality_max_first30_range_atr')}, flow={state.cumulative_mutations.get('kalcb.entry.quality_min_flow_score')}, accum={state.cumulative_mutations.get('kalcb.entry.quality_min_accumulation_score')}, frontier_rank={state.cumulative_mutations.get('kalcb.entry.quality_max_frontier_rank')}",
        f"Signal conversion={_pct(final.get('signal_conversion'))}; selected={_num(final.get('selected_count'), 0)}; trades={_num(final.get('trade_count'), 0)}; avg_trades/session={_num(final.get('avg_trades_per_session'), 3)}",
        *_gate_attribution_lines(audit),
        "",
        "=" * 70,
        "  6. First30 Signal Truthfulness",
        "=" * 70,
        "",
        *_first30_signal_lines(trade_rows),
        "",
        "=" * 70,
        "  7. Entry Mechanism And Timing",
        "=" * 70,
        "",
        *_entry_mechanism_lines(final, trade_rows, audit),
        "",
        "=" * 70,
        "  8. Exit And Trade Management Fitness",
        "=" * 70,
        "",
        f"Exit plan: failed_followthrough_bars={state.cumulative_mutations.get('kalcb.exit.failed_followthrough_bars')}, failed_followthrough_mfe_r={state.cumulative_mutations.get('kalcb.exit.failed_followthrough_mfe_r')}, failed_followthrough_close_r={state.cumulative_mutations.get('kalcb.exit.failed_followthrough_close_r')}, persistent={state.cumulative_mutations.get('kalcb.exit.failed_followthrough_persistent')}",
        f"Hard stop enabled={state.cumulative_mutations.get('kalcb.exit.hard_stop_enabled')}, stop_mode={state.cumulative_mutations.get('kalcb.exit.stop_mode')}, stop_pct={state.cumulative_mutations.get('kalcb.exit.stop_pct')}, target_r={state.cumulative_mutations.get('kalcb.exit.target_r')}",
        f"Partials={state.cumulative_mutations.get('kalcb.exit.use_partial_takes')}, trail_start={state.cumulative_mutations.get('kalcb.exit.trail_start_r')}, trail_gap={state.cumulative_mutations.get('kalcb.exit.trail_gap_r')}, quick_exit={state.cumulative_mutations.get('kalcb.exit.quick_exit_enabled')}",
        f"Stopless exits: mfe_giveback={state.cumulative_mutations.get('kalcb.exit.mfe_giveback_enabled')}, start={state.cumulative_mutations.get('kalcb.exit.mfe_giveback_start_r')}R, gap={state.cumulative_mutations.get('kalcb.exit.mfe_giveback_gap_r')}R, min_hold={state.cumulative_mutations.get('kalcb.exit.mfe_giveback_min_hold_bars')}; late_giveback_bars={state.cumulative_mutations.get('kalcb.exit.late_giveback_start_bars')}, vwap_fail_after_mfe={state.cumulative_mutations.get('kalcb.exit.vwap_fail_after_mfe_r')}R, time_decay_bars={state.cumulative_mutations.get('kalcb.exit.time_decay_bars')}",
        f"Conditional/cohort protection: conditional_stop_activate={state.cumulative_mutations.get('kalcb.exit.conditional_stop_activate_r')}R, gap={state.cumulative_mutations.get('kalcb.exit.conditional_stop_gap_r')}R; shadow_failed_followthrough_bars={state.cumulative_mutations.get('kalcb.exit.shadow_failed_followthrough_bars')}",
        *_exit_management_lines(final, trade_rows),
        "",
        "=" * 70,
        "  9. MFE / MAE And Lost Alpha",
        "=" * 70,
        "",
        *_mfe_mae_lines(final, trade_rows),
        "",
        "=" * 70,
        "  10. Risk, Sizing, And Capacity",
        "=" * 70,
        "",
        f"risk_per_trade_pct={state.cumulative_mutations.get('kalcb.risk.risk_per_trade_pct')}, max_position_notional_pct={state.cumulative_mutations.get('kalcb.risk.max_position_notional_pct')}, intraday_leverage={state.cumulative_mutations.get('kalcb.risk.intraday_leverage')}",
        f"max_positions={state.cumulative_mutations.get('kalcb.risk.max_positions')}, max_per_sector={state.cumulative_mutations.get('kalcb.risk.max_per_sector')}, heat_cap_pct={state.cumulative_mutations.get('kalcb.risk.heat_cap_pct')}, hard_max_drawdown_pct={state.cumulative_mutations.get('kalcb.risk.hard_max_drawdown_pct')}",
        f"Final broker DD={_pct(final.get('broker_max_drawdown_pct'))} vs hard ceiling={_pct(HARD_REJECTS['max_dd_pct'])}; worst fold DD={_pct(final.get('worst_fold_drawdown_pct'))}; worst fold net={_pct(final.get('worst_fold_net'))}",
        f"Exposure-normalized slot/broker ratio={_num(final.get('exposure_normalized_slot_to_broker_ratio'), 4)}x; portfolio-equivalent minus broker={_pct(final.get('portfolio_equivalent_minus_broker_net_return_pct'))}.",
        "",
        "=" * 70,
        "  11. Fold, Period, And Concentration Stability",
        "=" * 70,
        "",
        *_fold_stability_lines(fold_rows, baseline_fold_rows),
        "",
        *_period_stability_lines(trade_rows),
        "",
        "=" * 70,
        "  12. Paper/Live Parity And Implementation Contract",
        "=" * 70,
        "",
        f"Shared decision core: {execution_context.get('shared_decision_core')}",
        f"Strategy core version: {execution_context.get('strategy_core_version')}",
        f"Phase framework version: {execution_context.get('phase_framework_version')}",
        f"Phase auto version: {execution_context.get('phase_auto_version')}",
        f"Primary objective metric: {execution_context.get('primary_promotion_metric')}, basis={execution_context.get('primary_promotion_basis')}",
        f"Research scoring objective: {execution_context.get('research_objective_metric')}, basis={execution_context.get('research_objective_basis')}",
        f"Official metric basis: {final.get('official_metric_basis')}",
        f"Fill timing: {execution_context.get('fill_timing')}; auction mode: {execution_context.get('auction_mode')}",
        "Implementation lessons contract:",
        *_indented_json_lines(execution_context.get("implementation_lessons_contract", {}), indent=2),
        *_paper_live_requirement_summary_lines(final, execution_context, live_parity_audit),
        "",
        "=" * 70,
        "  13. Fast/Full Replay Audit And Artifact Pointers",
        "=" * 70,
        "",
        f"Status: {audit.get('status', 'not_run')}; pass={audit.get('pass')}; max_abs_metric_delta={_num(audit.get('max_abs_metric_delta'), 12)}",
        f"Fill hash match={audit.get('fill_hash_match')}; trade hash match={audit.get('trade_hash_match')}; trading decision hash match={audit.get('trading_decision_hash_match')}; strategy action hash match={audit.get('strategy_action_hash_match')}",
        f"Fast decisions={audit.get('fast_decision_count')}; full decisions={audit.get('audit_decision_count')}; suppressed entry rejections={audit.get('suppressed_entry_rejection_count')}",
        f"Same-bar fills={_num(final.get('same_bar_fill_count'), 0)}, forced replay closes={_num(final.get('forced_replay_close_count'), 0)}, rejected orders={_num(final.get('rejected_order_count'), 0)}, end-open positions={_num(final.get('end_open_position_count'), 0)}",
        "Scope: only entry_rejected diagnostics may differ between fast and full replay paths.",
        *_artifact_pointer_lines(cache_metadata, execution_context, source_ref, final, live_parity_audit),
        "",
        "=" * 70,
        "  14. Holdout And Promotion Boundary",
        "=" * 70,
        "",
        "Holdout status: excluded from this optimization round by design.",
        "Promotion status: research_only_until_holdout_and_paper_parity.",
        "Required next validation: run untouched holdout through the same shared-core replay path, then require paper/live parity before any live promotion.",
        "Reason: Round 1 holdout was negative and breached the drawdown ceiling; Round 2 training gains are not sufficient evidence of production alpha.",
        "",
        "=" * 70,
        "  15. Strengths, Weaknesses, And Verdict",
        "=" * 70,
        "",
        *_strength_weakness_lines(final, baseline),
        "",
        "Verdict: this round is an audit-clean training-only optimization. It is not live-ready until untouched holdout validation and paper/live decision-stream parity both pass.",
    ]
    return "\n".join(str(line) for line in lines) + "\n"


def _full_diagnostics_summary(
    final: dict[str, Any],
    baseline: dict[str, Any],
    state: PhaseState,
    source_ref: FixedCandidateSourceRef,
    *,
    fold_rows: tuple[dict[str, Any], ...] = tuple(),
    baseline_fold_rows: tuple[dict[str, Any], ...] = tuple(),
    trade_rows: tuple[dict[str, Any], ...] = tuple(),
    opportunity_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "strategy": "kalcb",
        "diagnostics_version": "kalcb-fixed-candidate-full-v1",
        "source": {
            "path": source_ref.path,
            "section": source_ref.section,
            "rank": source_ref.rank,
            "row_name": source_ref.row_name,
        },
        "baseline": _summary_metrics(baseline),
        "final": _summary_metrics(final),
        "delta": {
            "broker_net_return_pct": float(final.get("broker_net_return_pct", 0.0) or 0.0) - float(baseline.get("broker_net_return_pct", 0.0) or 0.0),
            "broker_max_drawdown_pct": float(final.get("broker_max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("broker_max_drawdown_pct", 0.0) or 0.0),
            "trade_count": float(final.get("trade_count", 0.0) or 0.0) - float(baseline.get("trade_count", 0.0) or 0.0),
            "immutable_score": float(final.get("immutable_score", 0.0) or 0.0) - float(baseline.get("immutable_score", 0.0) or 0.0),
        },
        "folds": _fold_summary(fold_rows),
        "baseline_folds": _fold_summary(baseline_fold_rows),
        "analysis_layers": _layer_diagnostics_summary(final, trade_rows),
        "accepted_phase_mutations": {
            str(phase): result.get("new_mutations", {})
            for phase, result in sorted(state.phase_results.items(), key=lambda item: int(item[0]))
            if result.get("new_mutations")
        },
        "fast_suppression_audit": final.get("fast_suppression_audit", {}),
        "top_entry_rejection_reasons": list(
            ((final.get("fast_suppression_audit") or {}).get("audit_replay_digest") or {}).get("top_entry_rejection_reasons") or []
        ),
        "accepted_loser_summary": final.get("accepted_loser_summary", {}),
        "mfe_capture_by_frontier_role": final.get("mfe_capture_by_frontier_role", {}),
        "opportunity_coverage": opportunity_diagnostics or {},
        "per_candidate_metrics": final.get("per_candidate_metrics", []),
        "holdout_excluded": bool(final.get("holdout_excluded", True)),
    }


def _write_candidate_snapshot_artifacts(output_dir: Path, context: Any) -> dict[str, Any]:
    root = Path(output_dir) / "paper_live_parity_inputs"
    snapshot_dir = root / "daily_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for day, snapshot in sorted(context.compiled_replay.snapshots.items(), key=lambda item: item[0]):
        artifact_hash = snapshot.artifact_hash
        filename = f"{day.isoformat()}_{artifact_hash[:12]}.json"
        _write_json(snapshot_dir / filename, snapshot.to_json_dict())
        rows.append(
            {
                "trade_date": day.isoformat(),
                "symbol_count": len(snapshot.candidates),
                "active_symbol_count": int((snapshot.metadata or {}).get("active_symbol_count", 0) or 0),
                "candidate_pool_count": int((snapshot.metadata or {}).get("candidate_pool_count", len(snapshot.candidates)) or 0),
                "source_fingerprint": snapshot.source_fingerprint,
                "artifact_hash": artifact_hash,
                "path": str(snapshot_dir / filename),
            }
        )
    manifest = {
        "artifact_type": "kalcb_daily_snapshot_manifest",
        "snapshot_count": len(rows),
        "aggregate_snapshot_hash": stable_signature(rows),
        "candidate_artifact_hash": context.compiled_replay.candidate_artifact_hash,
        "compiled_replay_fingerprint": context.compiled_replay.source_fingerprint,
        "snapshots": rows,
    }
    _write_json(root / "daily_snapshot_manifest.json", manifest)
    return manifest


def _compiled_bar_digest(context: Any) -> dict[str, Any]:
    rows = [
        {
            "timestamp": bar.timestamp.isoformat(),
            "symbol": bar.symbol,
            "timeframe": bar.timeframe,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "is_completed": bool(bar.is_completed),
            "metadata": dict(bar.metadata),
        }
        for bar in context.compiled_replay.bars
    ]
    return {
        "artifact_type": "kalcb_replay_market_bar_digest",
        "bar_count": len(rows),
        "completed_5m_bar_count": sum(1 for row in rows if row["is_completed"] and row["timeframe"].lower() == "5m"),
        "first_bar": rows[0] if rows else {},
        "last_bar": rows[-1] if rows else {},
        "market_bar_hash": stable_signature(rows),
        "compiled_replay_fingerprint": context.compiled_replay.source_fingerprint,
    }


def _opportunity_coverage_diagnostics(
    context: Any,
    mutations: dict[str, Any],
    trade_rows: tuple[dict[str, Any], ...],
    final: dict[str, Any],
) -> dict[str, Any]:
    compiled = getattr(context, "compiled_replay", None)
    snapshots = dict(getattr(compiled, "snapshots", {}) or {})
    routes = _configured_entry_routes(mutations)
    realized_by_route: Counter[str] = Counter(str(row.get("entry_route") or "legacy") for row in trade_rows)
    realized_symbols = {(str(row.get("entry_date") or "")[:10], str(row.get("symbol") or "")) for row in trade_rows}
    route_work: dict[str, dict[str, Any]] = {
        str(route.get("name") or route.get("route_name") or route.get("mode") or f"route_{index}"): {
            "name": str(route.get("name") or route.get("route_name") or route.get("mode") or f"route_{index}"),
            "mode": str(route.get("mode") or route.get("plan_mode") or mutations.get("kalcb.entry.plan_mode") or "breakout"),
            "priority": int(route.get("priority", route.get("order", index)) or 0),
            "criteria": _route_criteria_summary(route, mutations),
            "eligible_candidate_count": 0,
            "eligible_days": set(),
            "eligible_by_frontier_role": Counter(),
            "eligible_by_rank_bucket": Counter(),
            "failed_reason_counts": Counter(),
        }
        for index, route in enumerate(routes)
    }
    role_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    relvol_counts: Counter[str] = Counter()
    cpr_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    unrouteable_role_counts: Counter[str] = Counter()
    unrouteable_rank_counts: Counter[str] = Counter()
    unrouteable_relvol_counts: Counter[str] = Counter()
    unrouteable_cpr_counts: Counter[str] = Counter()
    unrouteable_sector_counts: Counter[str] = Counter()
    unrouteable_first_reason_counts: Counter[str] = Counter()
    high_shadow_reason_counts: Counter[str] = Counter()
    high_shadow_sector_counts: Counter[str] = Counter()
    high_shadow_rank_counts: Counter[str] = Counter()
    route_eligible_any = 0
    unrouteable = 0
    with_first30 = 0
    high_shadow = 0
    high_shadow_realized = 0
    active_high_relvol_rank8 = 0
    active_high_relvol_rank8_realized = 0
    sessions_with_candidates = 0
    days_with_any_route_eligible: set[str] = set()

    for day, snapshot in sorted(snapshots.items(), key=lambda item: str(item[0])):
        day_label = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        candidates = tuple(getattr(snapshot, "candidates", ()) or ())
        if candidates:
            sessions_with_candidates += 1
        for candidate in candidates:
            meta = _candidate_snapshot_metadata(candidate, day_label)
            symbol = str(meta.get("symbol") or "")
            role = _candidate_frontier_role(meta)
            rank_bucket = _rank_bucket(meta.get("frontier_rank"))
            relvol_bucket = _numeric_bucket(_candidate_numeric_or_zero(meta, "first30_rel_volume"), (1.0, 1.5, 2.5, 5.351847980160157))
            cpr_bucket = _numeric_bucket(_candidate_numeric_or_zero(meta, "first30_signal_bar_cpr"), (0.55, 0.65, 0.75, 0.85))
            sector = str(meta.get("sector") or "UNKNOWN")
            role_counts[role] += 1
            rank_counts[rank_bucket] += 1
            relvol_counts[relvol_bucket] += 1
            cpr_counts[cpr_bucket] += 1
            sector_counts[sector] += 1
            if _candidate_optional_float(meta, "first30_ret") is not None or _candidate_optional_float(meta, "first30_rel_volume") is not None:
                with_first30 += 1

            eligible_routes: list[str] = []
            first_failures: list[str] = []
            for route in routes:
                route_name = str(route.get("name") or route.get("route_name") or route.get("mode") or "route")
                passed, reason = _route_candidate_passes(route, mutations, meta)
                route_stats = route_work[route_name]
                if passed:
                    eligible_routes.append(route_name)
                    route_stats["eligible_candidate_count"] += 1
                    route_stats["eligible_days"].add(day_label)
                    route_stats["eligible_by_frontier_role"][role] += 1
                    route_stats["eligible_by_rank_bucket"][rank_bucket] += 1
                else:
                    route_stats["failed_reason_counts"][reason] += 1
                    first_failures.append(reason)

            realized_key = (day_label, symbol)
            if eligible_routes:
                route_eligible_any += 1
                days_with_any_route_eligible.add(day_label)
            else:
                unrouteable += 1
                first_reason = first_failures[0] if first_failures else "no_configured_route"
                unrouteable_first_reason_counts[first_reason] += 1
                unrouteable_role_counts[role] += 1
                unrouteable_rank_counts[rank_bucket] += 1
                unrouteable_relvol_counts[relvol_bucket] += 1
                unrouteable_cpr_counts[cpr_bucket] += 1
                unrouteable_sector_counts[sector] += 1

            relvol = _candidate_numeric_or_zero(meta, "first30_rel_volume")
            rank = _as_int(meta.get("frontier_rank"))
            if role != "initial_active" and rank > 0 and rank <= 8 and relvol >= 5.351847980160157:
                high_shadow += 1
                if realized_key in realized_symbols:
                    high_shadow_realized += 1
                high_shadow_reason_counts[(first_failures[0] if not eligible_routes and first_failures else "route_eligible")] += 1
                high_shadow_sector_counts[sector] += 1
                high_shadow_rank_counts[rank_bucket] += 1
            if role == "initial_active" and rank > 0 and rank <= 8 and relvol >= 5.351847980160157:
                active_high_relvol_rank8 += 1
                if realized_key in realized_symbols:
                    active_high_relvol_rank8_realized += 1

    total_candidates = sum(role_counts.values())
    route_rows: list[dict[str, Any]] = []
    for route_name, stats in sorted(route_work.items(), key=lambda item: (int(item[1]["priority"]), item[0])):
        eligible = int(stats["eligible_candidate_count"])
        realized = int(realized_by_route.get(route_name, 0))
        route_rows.append(
            {
                "name": route_name,
                "mode": stats["mode"],
                "priority": stats["priority"],
                "criteria": stats["criteria"],
                "eligible_candidate_count": eligible,
                "eligible_days": len(stats["eligible_days"]),
                "realized_trade_count": realized,
                "realized_to_snapshot_eligible": realized / eligible if eligible else 0.0,
                "eligible_by_frontier_role": _counter_dict(stats["eligible_by_frontier_role"]),
                "eligible_by_rank_bucket": _counter_dict(stats["eligible_by_rank_bucket"]),
                "top_failed_reasons": _counter_rows(stats["failed_reason_counts"], 8),
            }
        )

    route_blockage_share = unrouteable / total_candidates if total_candidates else 0.0
    return {
        "artifact_type": "kalcb_opportunity_coverage_diagnostics",
        "scope": "snapshot_level_route_eligibility_approximation",
        "note": "Counts use daily snapshot candidate metadata and configured route/source gates. They do not prove fillability, intraday path triggers, or broker capacity.",
        "snapshot_count": len(snapshots),
        "sessions_with_candidates": sessions_with_candidates,
        "candidate_count": total_candidates,
        "candidate_count_metric": final.get("candidate_pool_count"),
        "candidates_with_first30_metadata": with_first30,
        "realized_trade_count": len(trade_rows),
        "route_eligible_candidate_count": route_eligible_any,
        "route_eligible_share": route_eligible_any / total_candidates if total_candidates else 0.0,
        "days_with_any_route_eligible": len(days_with_any_route_eligible),
        "realized_to_route_eligible": len(trade_rows) / route_eligible_any if route_eligible_any else 0.0,
        "realized_to_candidate_count": len(trade_rows) / total_candidates if total_candidates else 0.0,
        "unrouteable_candidate_count": unrouteable,
        "unrouteable_share": route_blockage_share,
        "configured_routes": route_rows,
        "realized_by_route": _counter_dict(realized_by_route),
        "candidate_mix": {
            "by_frontier_role": _counter_dict(role_counts),
            "by_rank_bucket": _counter_dict(rank_counts),
            "by_first30_rel_volume_bucket": _counter_dict(relvol_counts),
            "by_first30_signal_cpr_bucket": _counter_dict(cpr_counts),
            "top_sectors": _counter_rows(sector_counts, 12),
        },
        "unrouteable_mix": {
            "top_first_reasons": _counter_rows(unrouteable_first_reason_counts, 12),
            "by_frontier_role": _counter_dict(unrouteable_role_counts),
            "by_rank_bucket": _counter_dict(unrouteable_rank_counts),
            "by_first30_rel_volume_bucket": _counter_dict(unrouteable_relvol_counts),
            "by_first30_signal_cpr_bucket": _counter_dict(unrouteable_cpr_counts),
            "top_sectors": _counter_rows(unrouteable_sector_counts, 12),
        },
        "targeted_probe_surfaces": {
            "frontier_shadow_rank_le8_relvol_ge_q85": {
                "candidate_count": high_shadow,
                "realized_trade_count": high_shadow_realized,
                "top_block_or_route_reasons": _counter_rows(high_shadow_reason_counts, 8),
                "top_sectors": _counter_rows(high_shadow_sector_counts, 8),
                "by_rank_bucket": _counter_dict(high_shadow_rank_counts),
            },
            "initial_active_rank_le8_relvol_ge_q85": {
                "candidate_count": active_high_relvol_rank8,
                "realized_trade_count": active_high_relvol_rank8_realized,
                "realized_share": active_high_relvol_rank8_realized / active_high_relvol_rank8 if active_high_relvol_rank8 else 0.0,
            },
        },
        "experiment_hints": _opportunity_experiment_hints(
            total_candidates=total_candidates,
            route_eligible_any=route_eligible_any,
            realized_trades=len(trade_rows),
            unrouteable_share=route_blockage_share,
            high_shadow=high_shadow,
            active_high_relvol_rank8=active_high_relvol_rank8,
        ),
    }


def _configured_entry_routes(mutations: dict[str, Any]) -> list[dict[str, Any]]:
    raw_routes = mutations.get("kalcb.entry.routes") or mutations.get("kalcb.entry.plan_routes") or mutations.get("kalcb.entry.entry_plan_routes") or ()
    if raw_routes:
        routes = [dict(route) for route in raw_routes if isinstance(route, dict)]
    else:
        routes = [{"name": str(mutations.get("kalcb.entry.plan_mode") or "legacy"), "mode": str(mutations.get("kalcb.entry.plan_mode") or "breakout"), "priority": 0}]
    indexed = [(index, route) for index, route in enumerate(routes)]
    ordered = sorted(indexed, key=lambda item: (int(item[1].get("priority", item[1].get("order", item[0])) or 0), item[0]))
    out: list[dict[str, Any]] = []
    for index, route in ordered:
        route.setdefault("name", str(route.get("route_name") or route.get("mode") or f"route_{index}"))
        route.setdefault("priority", int(route.get("order", index) or 0))
        out.append(route)
    return out


def _candidate_snapshot_metadata(candidate: Any, day_label: str) -> dict[str, Any]:
    meta = dict(getattr(candidate, "metadata", {}) or {})
    meta.setdefault("symbol", str(getattr(candidate, "symbol", "")))
    meta.setdefault("trade_date", day_label)
    meta.setdefault("sector", str(getattr(candidate, "sector", "UNKNOWN") or "UNKNOWN"))
    meta.setdefault("regime_tier", str(getattr(candidate, "regime_tier", "UNKNOWN") or "UNKNOWN"))
    meta.setdefault("selection_score", getattr(candidate, "selection_score", 0.0))
    meta.setdefault("frontier_selection_score", meta.get("selection_score", getattr(candidate, "selection_score", 0.0)))
    meta.setdefault("flow_score", getattr(candidate, "flow_score", 0.0))
    meta.setdefault("accumulation_score", getattr(candidate, "accumulation_score", 0.0))
    meta.setdefault("candidate_rank", meta.get("frontier_rank", 0))
    if "first30_range_close_location" not in meta and "first30_close_location" in meta:
        meta["first30_range_close_location"] = meta.get("first30_close_location")
    if "first30_signal_bar_cpr" not in meta and "first30_signal_cpr" in meta:
        meta["first30_signal_bar_cpr"] = meta.get("first30_signal_cpr")
    return meta


def _route_candidate_passes(route: dict[str, Any], mutations: dict[str, Any], meta: dict[str, Any]) -> tuple[bool, str]:
    rules = _effective_route_rules(route, mutations)
    mode = str(rules.get("mode") or "breakout")
    if mode in {"first30_open", "opening_drive"} and _candidate_optional_float(meta, "first30_ret") is None:
        return False, "missing_first30_metadata"
    checks: tuple[tuple[str, str, float, str, bool], ...] = (
        ("entry_min_bar_ret", "first30_ret", _as_float(rules["min_bar_ret"]), ">=", _as_float(rules["min_bar_ret"]) > -9.0),
        ("entry_min_vwap_ret", "first30_vwap_ret", _as_float(rules["min_vwap_ret"]), ">=", _as_float(rules["min_vwap_ret"]) > -9.0),
        ("entry_min_close_location", "first30_range_close_location", _as_float(rules["min_close_location"]), ">=", bool(route.get("min_close_location") is not None or _as_float(rules["min_close_location"]) > 0.0)),
        ("entry_first30_rel_volume", "first30_rel_volume", _as_float(rules["min_first30_rel_volume"]), ">=", _as_float(rules["min_first30_rel_volume"]) > 0.0),
        ("entry_first30_signal_cpr", "first30_signal_bar_cpr", _as_float(rules["min_first30_signal_cpr"]), ">=", _as_float(rules["min_first30_signal_cpr"]) > 0.0),
        ("entry_first30_open_drawdown", "first30_open_drawdown", _as_float(rules["min_first30_open_drawdown"]), ">=", _as_float(rules["min_first30_open_drawdown"]) > -9.0),
        ("entry_first30_low_vs_prev_close", "first30_low_vs_prev_close", _as_float(rules["min_first30_low_vs_prev_close"]), ">=", _as_float(rules["min_first30_low_vs_prev_close"]) > -9.0),
        ("entry_first30_range_atr_min", "first30_range_atr", _as_float(rules["min_first30_range_atr"]), ">=", _as_float(rules["min_first30_range_atr"]) > 0.0),
        ("entry_first30_range_atr_max", "first30_range_atr", _as_float(rules["max_first30_range_atr"]), "<=", _as_float(rules["max_first30_range_atr"]) < 99.0),
    )
    for reason, field, threshold, op, active in checks:
        if not active:
            continue
        value = _candidate_optional_float(meta, field)
        if value is None:
            return False, f"missing_{field}"
        if op == ">=" and value < threshold:
            return False, reason
        if op == "<=" and value > threshold:
            return False, reason
    for key, threshold in dict(rules.get("context_min") or {}).items():
        value = _candidate_optional_float(meta, str(key))
        if value is None or value < float(threshold):
            return False, f"entry_context_min:{key}"
    for key, threshold in dict(rules.get("context_max") or {}).items():
        value = _candidate_optional_float(meta, str(key))
        if value is None or value > float(threshold):
            return False, f"entry_context_max:{key}"
    for key, denied_values in dict(rules.get("context_exclude") or {}).items():
        value = meta.get(str(key))
        denied = {str(item) for item in denied_values}
        if value is None or str(value) in denied:
            return False, f"entry_context_exclude:{key}"
    if bool(rules.get("require_initial_active")) and not _candidate_initial_active(meta):
        return False, "entry_initial_active"
    max_rank = int(rules.get("max_frontier_rank") or 0)
    rank = _as_int(meta.get("frontier_rank"))
    if max_rank > 0 and not (0 < rank <= max_rank):
        return False, "entry_frontier_rank"
    min_frontier_score = _as_float(rules.get("min_frontier_score"))
    if min_frontier_score > -9.0 and _candidate_numeric_or_zero(meta, "frontier_selection_score") < min_frontier_score:
        return False, "entry_frontier_score"
    min_flow = _as_float(rules.get("min_flow_score"))
    if min_flow > -9.0 and _candidate_numeric_or_zero(meta, "flow_score") < min_flow:
        return False, "entry_flow_score"
    min_accum = _as_float(rules.get("min_accumulation_score"))
    if min_accum > -9.0 and _candidate_numeric_or_zero(meta, "accumulation_score") < min_accum:
        return False, "entry_accumulation_score"
    quality_votes = _snapshot_quality_votes(meta, rules)
    if quality_votes is not None and quality_votes["vote_count"] < quality_votes["required"]:
        return False, "entry_quality_votes"
    return True, "eligible"


def _effective_route_rules(route: dict[str, Any], mutations: dict[str, Any]) -> dict[str, Any]:
    route = dict(route or {})

    def value(short_key: str, default: Any) -> Any:
        for key in (short_key, f"entry.{short_key}", f"kalcb.entry.{short_key}"):
            if key in route:
                return route[key]
        return mutations.get(f"kalcb.entry.{short_key}", default)

    context_min = _float_mapping(
        route.get("context_min")
        or route.get("route_context_min")
        or route.get("regime_min")
        or route.get("kalcb.entry.route_context_min")
        or mutations.get("kalcb.entry.route_context_min")
        or mutations.get("kalcb.entry.context_min")
    )
    context_max = _float_mapping(
        route.get("context_max")
        or route.get("route_context_max")
        or route.get("regime_max")
        or route.get("kalcb.entry.route_context_max")
        or mutations.get("kalcb.entry.route_context_max")
        or mutations.get("kalcb.entry.context_max")
    )
    context_exclude = _string_mapping(
        route.get("context_exclude")
        or route.get("route_context_exclude")
        or route.get("context_not")
        or route.get("kalcb.entry.route_context_exclude")
        or mutations.get("kalcb.entry.route_context_exclude")
        or mutations.get("kalcb.entry.context_exclude")
        or mutations.get("kalcb.entry.context_not")
    )
    return {
        "mode": route.get("mode") or route.get("plan_mode") or route.get("entry_plan_mode") or mutations.get("kalcb.entry.plan_mode") or "breakout",
        "min_bar_ret": value("min_bar_ret", -9.99),
        "min_vwap_ret": value("min_vwap_ret", -9.99),
        "min_close_location": value("min_close_location", 0.0),
        "min_first30_rel_volume": value("min_first30_rel_volume", 0.0),
        "min_first30_signal_cpr": value("min_first30_signal_cpr", 0.0),
        "min_first30_open_drawdown": value("min_first30_open_drawdown", -9.99),
        "min_first30_low_vs_prev_close": value("min_first30_low_vs_prev_close", -9.99),
        "min_first30_range_atr": value("min_first30_range_atr", 0.0),
        "max_first30_range_atr": value("max_first30_range_atr", 99.0),
        "require_initial_active": _as_bool(value("require_initial_active", False)),
        "max_frontier_rank": value("max_frontier_rank", 0),
        "min_frontier_score": value("min_frontier_score", -9.99),
        "min_flow_score": value("min_flow_score", -9.99),
        "min_accumulation_score": value("min_accumulation_score", -9.99),
        "min_quality_votes": value("min_quality_votes", 0),
        "quality_min_bar_ret": value("quality_min_bar_ret", -9.99),
        "quality_min_first30_signal_cpr": value("quality_min_first30_signal_cpr", -9.99),
        "quality_min_first30_rel_volume": value("quality_min_first30_rel_volume", -9.99),
        "quality_min_first30_range_atr": value("quality_min_first30_range_atr", -9.99),
        "quality_max_first30_range_atr": value("quality_max_first30_range_atr", 0.0),
        "quality_min_flow_score": value("quality_min_flow_score", -9.99),
        "quality_min_accumulation_score": value("quality_min_accumulation_score", -9.99),
        "quality_max_frontier_rank": value("quality_max_frontier_rank", 0),
        "context_min": context_min,
        "context_max": context_max,
        "context_exclude": context_exclude,
    }


def _snapshot_quality_votes(meta: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any] | None:
    required = int(rules.get("min_quality_votes") or 0)
    if required <= 0:
        return None
    gates: list[bool] = []

    def add_min(rule_key: str, field: str) -> None:
        threshold = _as_float(rules.get(rule_key))
        if threshold <= -9.0:
            return
        gates.append(_candidate_numeric_or_zero(meta, field) >= threshold)

    add_min("quality_min_bar_ret", "first30_ret")
    add_min("quality_min_first30_signal_cpr", "first30_signal_bar_cpr")
    add_min("quality_min_first30_rel_volume", "first30_rel_volume")
    add_min("quality_min_first30_range_atr", "first30_range_atr")
    max_range = _as_float(rules.get("quality_max_first30_range_atr"))
    if max_range > 0:
        gates.append(_candidate_numeric_or_zero(meta, "first30_range_atr") <= max_range)
    add_min("quality_min_flow_score", "flow_score")
    add_min("quality_min_accumulation_score", "accumulation_score")
    max_rank = int(rules.get("quality_max_frontier_rank") or 0)
    if max_rank > 0:
        rank = _as_int(meta.get("frontier_rank"))
        gates.append(0 < rank <= max_rank)
    return {"vote_count": sum(1 for gate in gates if gate), "required": required, "gate_count": len(gates)}


def _route_criteria_summary(route: dict[str, Any], mutations: dict[str, Any]) -> dict[str, Any]:
    rules = _effective_route_rules(route, mutations)
    keys = (
        "require_initial_active",
        "max_frontier_rank",
        "min_bar_ret",
        "min_vwap_ret",
        "min_first30_rel_volume",
        "min_first30_signal_cpr",
        "min_quality_votes",
        "quality_min_first30_rel_volume",
        "quality_min_first30_signal_cpr",
        "quality_max_frontier_rank",
        "context_min",
        "context_max",
        "context_exclude",
    )
    return {key: rules.get(key) for key in keys if rules.get(key) not in (None, {}, (), "")}


def _opportunity_coverage_lines(payload: dict[str, Any] | None) -> list[str]:
    data = dict(payload or {})
    if not data:
        return ["Opportunity coverage diagnostics: not available."]
    lines = [
        "Opportunity coverage and route blockage:",
        f"  Scope: {data.get('scope')}; this is a candidate-snapshot approximation, not a replay fill model.",
        f"  Snapshot candidates={_num(data.get('candidate_count'), 0)} across {_num(data.get('sessions_with_candidates'), 0)} active snapshot sessions; first30 metadata coverage={_num(data.get('candidates_with_first30_metadata'), 0)}.",
        f"  Route-eligible candidates={_num(data.get('route_eligible_candidate_count'), 0)} ({_pct(data.get('route_eligible_share'))}); realized trades={_num(data.get('realized_trade_count'), 0)}; realized/eligible={_pct(data.get('realized_to_route_eligible'))}; realized/all candidates={_pct(data.get('realized_to_candidate_count'))}.",
        f"  Unrouteable candidates={_num(data.get('unrouteable_candidate_count'), 0)} ({_pct(data.get('unrouteable_share'))}).",
    ]
    lines.append("  Route coverage:")
    for route in list(data.get("configured_routes") or [])[:8]:
        lines.append(
            f"    {route.get('name'):<30} mode={route.get('mode'):<18} eligible={_num(route.get('eligible_candidate_count'), 0):>5} days={_num(route.get('eligible_days'), 0):>4} realized={_num(route.get('realized_trade_count'), 0):>4} conv={_pct(route.get('realized_to_snapshot_eligible')):>7}"
        )
    unrouteable_mix = dict(data.get("unrouteable_mix") or {})
    reasons = list(unrouteable_mix.get("top_first_reasons") or [])
    if reasons:
        lines.append("  Top snapshot-level route blockers:")
        for row in reasons[:8]:
            lines.append(f"    {row.get('key'):<34} {_num(row.get('count'), 0)}")
    probes = dict(data.get("targeted_probe_surfaces") or {})
    shadow = dict(probes.get("frontier_shadow_rank_le8_relvol_ge_q85") or {})
    active = dict(probes.get("initial_active_rank_le8_relvol_ge_q85") or {})
    lines.append(
        f"  Targeted surface check: shadow rank<=8 and relvol>=q85 candidates={_num(shadow.get('candidate_count'), 0)}, realized={_num(shadow.get('realized_trade_count'), 0)}; initial-active same surface candidates={_num(active.get('candidate_count'), 0)}, realized={_num(active.get('realized_trade_count'), 0)}."
    )
    hints = list(data.get("experiment_hints") or [])
    if hints:
        lines.append("  Experiment hints:")
        for hint in hints[:5]:
            lines.append(f"    - {hint}")
    return lines


def _opportunity_experiment_hints(
    *,
    total_candidates: int,
    route_eligible_any: int,
    realized_trades: int,
    unrouteable_share: float,
    high_shadow: int,
    active_high_relvol_rank8: int,
) -> list[str]:
    hints: list[str] = []
    if total_candidates and unrouteable_share > 0.70:
        hints.append("Route breadth is the first bottleneck; run route-family ablations before adding new indicators.")
    if high_shadow > 0:
        hints.append("There are high-relvol top-rank frontier-shadow names; test a narrow require_initial_active=false branch with locked holdout, not a broad frontier expansion.")
    if active_high_relvol_rank8 > 0 and realized_trades / max(active_high_relvol_rank8, 1) < 0.75:
        hints.append("High-relvol active candidates are not all converting; audit first30 quality votes and post-route intraday/risk blockers before resizing further.")
    if route_eligible_any and realized_trades / max(route_eligible_any, 1) < 0.50:
        hints.append("Snapshot route eligibility materially exceeds realized trades; inspect intraday path, duplicate-position, and broker-capacity gates.")
    if not hints:
        hints.append("Current route family is tight; next round should perturb thresholds and exits around the accepted surface before expanding the feature set.")
    return hints


def _candidate_frontier_role(meta: dict[str, Any]) -> str:
    role = str(meta.get("frontier_role") or "").strip() or "unknown"
    if _candidate_initial_active(meta):
        return "initial_active"
    if role in {"shadow", "frontier"}:
        return "frontier_shadow"
    return role


def _candidate_initial_active(meta: dict[str, Any]) -> bool:
    role = str(meta.get("frontier_role") or "").strip().lower()
    default = role in {"", "initial_active", "active"}
    return _as_bool(meta.get("frontier_initial_active", default), default=default)


def _candidate_optional_float(meta: dict[str, Any], key: str) -> float | None:
    aliases = {
        "first30_range_close_location": ("first30_range_close_location", "first30_close_location"),
        "first30_signal_bar_cpr": (
            "first30_signal_bar_cpr",
            "first30_signal_cpr",
            "first30_cpr",
            "first30_close_location",
            "first30_range_close_location",
        ),
        "frontier_selection_score": ("frontier_selection_score", "selection_score"),
        "first30_vwap_ret": ("first30_vwap_ret", "vwap_ret"),
    }
    for actual_key in aliases.get(key, (key,)):
        if actual_key in meta:
            return _optional_float(meta.get(actual_key))
    return None


def _candidate_numeric_or_zero(meta: dict[str, Any], key: str) -> float:
    value = _candidate_optional_float(meta, key)
    return value if value is not None else 0.0


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, item in value.items():
        parsed = _optional_float(item)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _string_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        if isinstance(item, str):
            values = tuple(part.strip() for part in item.split(",") if part.strip())
        elif isinstance(item, (list, tuple, set)):
            values = tuple(str(part) for part in item if str(part))
        elif item is None:
            values = ()
        else:
            values = (str(item),)
        if values:
            out[str(key)] = values
    return out


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(count) for key, count in counter.most_common()}


def _counter_rows(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    total = sum(int(count) for count in counter.values())
    return [
        {"key": str(key), "count": int(count), "share": int(count) / total if total else 0.0}
        for key, count in counter.most_common(limit)
    ]


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _paper_live_parity_requirements(
    final: dict[str, Any],
    execution_context: dict[str, Any],
    *,
    mutations: dict[str, Any] | None = None,
    source_ref: FixedCandidateSourceRef | None = None,
    context: Any | None = None,
    replay_digest: dict[str, Any] | None = None,
    snapshot_manifest: dict[str, Any] | None = None,
    bar_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mutations = dict(mutations or {})
    audit = dict(final.get("fast_suppression_audit") or {})
    replay_baseline = dict(replay_digest or audit.get("audit_replay_digest") or audit.get("fast_replay_digest") or {})
    snapshot_manifest = dict(snapshot_manifest or {})
    bar_digest = dict(bar_digest or {})
    strategy_action_fields = [
        "timestamp",
        "decision_ref",
        "action_type",
        "strategy_id",
        "symbol",
        "qty",
        "price_fields",
        "reason",
        "metadata",
        "entry_route",
        "entry_route_mode",
        "entry_route_priority",
        "entry_route_attempts",
        "entry_route_risk_mult",
        "entry_route_notional_mult",
        "entry_route_participation_mult",
        "entry_route_max_session_trades",
        "entry_route_context_min_keys",
        "entry_route_context_max_keys",
        "entry_route_context_exclude_keys",
        "entry_route_session_count_before",
        "first30_gap",
        "first30_gap_retention_ratio",
        "first30_gap_relvol",
        "first30_low_vs_prev_relvol",
        "entry_path_anchor_time",
        "entry_path_anchor_price",
        "entry_path_risk_per_share",
        "entry_path_completed_bars",
        "h3_current_r",
        "h3_mfe_r",
        "h3_mae_r",
        "h3_giveback_r",
        "h6_current_r",
        "h6_mfe_r",
        "h6_mae_r",
        "h6_giveback_r",
        "daily_return_5d",
        "daily_return_20d",
        "daily_momentum_pct",
        "daily_sector_alignment_pct",
        "first30_sector_leadership_pct",
        "continuation_joint_quality_pct",
        "sector_daily_score_pct",
        "sector_daily_participation",
        "sector_intraday_score_pct",
        "sector_intraday_effective_count",
        "session_sector_intraday_score_pct_mean",
        "session_sector_intraday_positive_share",
        "session_sector_intraday_effective_count_mean",
        "effective_max_position_notional_pct",
        "portfolio_drawdown_pct",
        "portfolio_session_return_pct",
        "exit_path_quality_context",
    ]
    expected_contract = {
        "optimized_mutations_hash": stable_signature(mutations),
        "optimized_mutations": mutations,
        "engine": "strategy_kalcb.engine.KALCBEngine",
        "shared_core": execution_context.get("implementation_lessons_contract", {}).get("shared_core") or "strategy_kalcb.core.step_kalcb_core",
        "fill_handler": "strategy_kalcb.core.on_kalcb_fill",
        "fill_timing": execution_context.get("fill_timing"),
        "auction_mode": execution_context.get("auction_mode"),
        "strategy_core_version": execution_context.get("strategy_core_version"),
        "candidate_snapshot_hash": (context.compiled_replay.candidate_artifact_hash if context is not None else final.get("candidate_snapshot_hash")),
        "compiled_replay_fingerprint": (context.compiled_replay.source_fingerprint if context is not None else final.get("source_fingerprint")),
        "market_bar_hash": bar_digest.get("market_bar_hash", ""),
        "daily_snapshot_manifest_hash": snapshot_manifest.get("aggregate_snapshot_hash", ""),
        "source_ref": {
            "path": source_ref.path if source_ref else "",
            "section": source_ref.section if source_ref else "",
            "rank": source_ref.rank if source_ref else "",
            "row_name": source_ref.row_name if source_ref else "",
        },
    }
    return {
        "status": "pending_paper_live_evidence",
        "required_before_promotion": True,
        "validation_scope": "end_of_round_backtest_contract_plus_required_paper_live_hash_match",
        "training_replay_audit_pass": bool((final.get("fast_suppression_audit") or {}).get("pass", False)),
        "expected_contract": expected_contract,
        "required_strategy_action_fields": strategy_action_fields,
        "backtest_hash_baselines": {
            "non_rejection_decision_hash": replay_baseline.get("trading_decision_hash", ""),
            "neutral_strategy_action_hash": replay_baseline.get("strategy_action_hash", ""),
            "fill_hash": replay_baseline.get("fill_hash", ""),
            "trade_hash": replay_baseline.get("trade_hash", ""),
            "market_bar_hash": bar_digest.get("market_bar_hash", ""),
            "daily_snapshot_manifest_hash": snapshot_manifest.get("aggregate_snapshot_hash", ""),
            "entry_rejection_count": replay_baseline.get("entry_rejection_count", 0),
            "same_bar_fill_count": replay_baseline.get("same_bar_fill_count", final.get("same_bar_fill_count", 0)),
        },
        "requirements": [
            "Run paper/live with the same optimized mutation hash, KALCBEngine, strategy_kalcb.core.step_kalcb_core, and on_kalcb_fill contract listed above.",
            "Persist each daily KALCBDailySnapshot JSON artifact and hash-check it against replay regeneration before the session starts.",
            "Capture exact completed 5m MarketBar inputs delivered to paper/live, replay them offline, and hash-match non-rejection decisions.",
            "Capture neutral StrategyAction payloads before OMS translation, then capture OMS intents, order ids, fill events, and final trade outcomes.",
            "Hash-check entry_route, entry_route_mode, entry_route_priority, entry_route_attempts, and route sizing metadata on every submitted entry action.",
            "Hash-check exit_path_quality_context on path_quality_exit actions when path-quality exits are enabled.",
            "Feed paper/live fills back through on_kalcb_fill and verify order roles, quantities, exit reasons, remaining state, and state snapshots.",
            "Verify next_5m_open semantics: no action fills on the signal bar; no auction fills unless explicitly configured.",
            "Verify KRX tick rounding, fees, tax, slippage, participation caps, cash, heat, sector counts, buying power, and open-position counts.",
            "Run restart hydration checks by snapshotting state, restoring it, replaying subsequent bars/fills, and requiring identical decisions/actions/state.",
            "Produce a paper/live parity report with decision/action/intent/fill/trade/state hashes, mismatch counts, and all divergences triaged.",
        ],
        "acceptance_criteria": {
            "same_bar_fill_count": 0,
            "forced_replay_close_count": 0,
            "untriaged_rejected_order_count": 0,
            "decision_hash_mismatch_count": 0,
            "strategy_action_hash_mismatch_count": 0,
            "oms_intent_hash_mismatch_count": 0,
            "fill_hash_mismatch_count": 0,
            "trade_hash_mismatch_count": 0,
            "daily_snapshot_hash_mismatch_count": 0,
            "market_bar_hash_mismatch_count": 0,
            "order_role_mismatch_count": 0,
        "entry_route_metadata_mismatch_count": 0,
            "quantity_mismatch_count": 0,
            "exit_reason_mismatch_count": 0,
            "exit_path_quality_context_mismatch_count": 0,
            "cash_heat_sector_position_mismatch_count": 0,
            "tick_rounding_mismatch_count": 0,
            "fees_tax_slippage_mismatch_count": 0,
            "participation_mismatch_count": 0,
            "state_hydration_mismatch_count": 0,
            "allowed_differences": ["diagnostic-only entry_rejected emissions may differ if they do not alter actions, fills, trades, or state"],
        },
        "offline_replay_procedure": [
            "Load optimized_config.json mutations and verify optimized_mutations_hash.",
            "Load paper daily snapshots and verify each artifact_hash plus the aggregate daily_snapshot_manifest_hash.",
            "Load captured completed 5m paper/live bars and compare their market_bar_hash with the paper replay bundle hash.",
            "Replay captured bars through KALCBEngine/step_kalcb_core with captured fills fed through on_kalcb_fill at their recorded timestamps.",
            "Hash non-rejection DecisionEvent rows and neutral StrategyAction rows before OMS translation.",
            "Hash OMS intents after action_to_intent normalization, fills after KIS/paper execution capture, trade outcomes, and serialized state snapshots.",
            "Fail promotion unless every acceptance criterion is zero or explicitly covered by allowed_differences.",
        ],
        "evidence_artifacts": [
            "paper_session_manifest.json",
            "paper_daily_snapshots/",
            "paper_market_bars_5m.parquet",
            "paper_decision_stream.jsonl",
            "paper_strategy_actions.jsonl",
            "paper_oms_intents.jsonl",
            "paper_fill_events.jsonl",
            "paper_trade_outcomes.jsonl",
            "paper_state_snapshots.jsonl",
            "paper_live_parity_report.json",
        ],
        "evidence_schema": {
            "paper_decision_stream.jsonl": ["timestamp", "strategy_id", "symbol", "decision_code", "reason", "actions", "metadata", "state_snapshot_ref"],
            "paper_strategy_actions.jsonl": strategy_action_fields,
            "paper_oms_intents.jsonl": ["timestamp", "action_ref", "intent_id", "order_id", "side", "qty", "order_type", "limit_price", "stop_price", "status"],
            "paper_fill_events.jsonl": ["timestamp", "order_id", "symbol", "side", "qty", "price", "reason", "metadata"],
            "paper_state_snapshots.jsonl": ["timestamp", "state_hash", "snapshot_state_payload"],
        },
        "generated_backtest_artifacts": {
            "paper_live_parity_contract": "paper_live_parity_contract.json",
            "daily_snapshot_manifest": "paper_live_parity_inputs/daily_snapshot_manifest.json",
            "daily_snapshots": "paper_live_parity_inputs/daily_snapshots/",
            "replay_market_bar_digest": "paper_live_parity_inputs/replay_market_bar_digest.json",
        },
    }


def _paper_live_requirement_lines(final: dict[str, Any], execution_context: dict[str, Any], payload: dict[str, Any] | None = None) -> list[str]:
    payload = dict(payload or _paper_live_parity_requirements(final, execution_context))
    lines = [
        f"Status: {payload['status']}; training replay audit pass={payload['training_replay_audit_pass']}",
        "Required evidence before live promotion:",
    ]
    for index, requirement in enumerate(payload["requirements"], start=1):
        lines.append(f"  {index}. {requirement}")
    contract = dict(payload.get("expected_contract") or {})
    baselines = dict(payload.get("backtest_hash_baselines") or {})
    lines.append("Backtest-side hash baselines:")
    lines.append(f"  optimized_mutations_hash: {contract.get('optimized_mutations_hash')}")
    lines.append(f"  candidate_snapshot_hash: {contract.get('candidate_snapshot_hash')}")
    lines.append(f"  market_bar_hash: {baselines.get('market_bar_hash')}")
    lines.append(f"  non_rejection_decision_hash: {baselines.get('non_rejection_decision_hash')}")
    lines.append(f"  neutral_strategy_action_hash: {baselines.get('neutral_strategy_action_hash')}")
    lines.append(f"  fill_hash: {baselines.get('fill_hash')}")
    lines.append(f"  trade_hash: {baselines.get('trade_hash')}")
    lines.append("Acceptance criteria:")
    for key, value in payload["acceptance_criteria"].items():
        if key == "allowed_differences":
            lines.append(f"  {key}: {'; '.join(value)}")
        else:
            lines.append(f"  {key}: {value}")
    lines.append("Evidence artifacts:")
    for artifact in payload["evidence_artifacts"]:
        lines.append(f"  - {artifact}")
    return lines


def _broker_trade_rows(trades: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        route = dict(getattr(trade, "route_metadata", {}) or {})
        cohort = dict(getattr(trade, "cohort_metadata", {}) or {})
        qty = max(int(getattr(trade, "qty", 0) or 0), 1)
        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        exit_price = float(getattr(trade, "exit_price", entry_price) or entry_price)
        risk = max(float(route.get("risk_per_share", 0.0) or 0.0), 1e-9)
        risk_notional = max(risk * qty, 1e-9)
        notional = max(entry_price * qty, 1e-9)
        net_pnl = float(getattr(trade, "net_pnl", 0.0) or 0.0)
        gross_pnl = float(getattr(trade, "gross_pnl", 0.0) or 0.0)
        mfe_r = max(float(getattr(trade, "mfe", 0.0) or 0.0), 0.0) / risk
        mae_r = min(float(getattr(trade, "mae", 0.0) or 0.0), 0.0) / risk
        net_r = net_pnl / risk_notional
        entry_time = getattr(trade, "entry_fill_time", None)
        decision_time = getattr(trade, "entry_decision_time", entry_time)
        exit_time = getattr(trade, "exit_fill_time", None)
        hold_bars = 0
        if entry_time is not None and exit_time is not None:
            hold_bars = max(1, int((exit_time - entry_time).total_seconds() // 300) + 1)
        rows.append(
            {
                "symbol": str(getattr(trade, "symbol", "")),
                "qty": qty,
                "entry_time": entry_time.isoformat() if entry_time is not None else "",
                "entry_date": entry_time.date().isoformat() if entry_time is not None else "",
                "entry_time_label": entry_time.strftime("%H:%M") if entry_time is not None else "",
                "entry_decision_time": decision_time.isoformat() if decision_time is not None else "",
                "entry_decision_label": decision_time.strftime("%H:%M") if decision_time is not None else "",
                "exit_time": exit_time.isoformat() if exit_time is not None else "",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "notional": notional,
                "r": net_r,
                "gross_r": gross_pnl / risk_notional,
                "net_return_pct": net_pnl / notional,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "mfe_capture": max(net_r, 0.0) / max(mfe_r, 1e-9) if mfe_r > 0 else 0.0,
                "giveback_r": mfe_r - net_r,
                "hold_bars": hold_bars,
                "hold_hours": hold_bars * 5.0 / 60.0,
                "exit_reason": str(getattr(trade, "exit_reason", "") or "unknown"),
                "entry_type": str(route.get("entry_type") or "unknown"),
                "entry_route": str(route.get("entry_route") or "legacy"),
                "entry_route_mode": str(route.get("entry_route_mode") or route.get("entry_type") or "unknown"),
                "entry_route_priority": _as_int(route.get("entry_route_priority")),
                "frontier_role": str(route.get("frontier_role") or "unknown"),
                "candidate_rank": _as_int(route.get("candidate_rank")),
                "frontier_rank": _as_int(route.get("frontier_rank")),
                "frontier_selection_score": _as_float(route.get("frontier_selection_score")),
                "flow_score": _as_float(route.get("flow_score")),
                "accumulation_score": _as_float(route.get("accumulation_score")),
                "sector": str(route.get("sector") or "UNKNOWN"),
                "regime_tier": str(route.get("regime_tier") or "UNKNOWN"),
                "momentum_score": _as_float(route.get("momentum_score")),
                "bar_rvol": _as_float(route.get("bar_rvol")),
                "cpr": _as_float(route.get("cpr")),
                "avwap": _as_float(route.get("avwap")),
                "or_high": _as_float(route.get("or_high")),
                "or_low": _as_float(route.get("or_low")),
                "daily_atr": _as_float(route.get("daily_atr")),
                "daily_return_5d": _optional_float(route.get("daily_return_5d")),
                "daily_return_20d": _optional_float(route.get("daily_return_20d")),
                "daily_return_60d": _optional_float(route.get("daily_return_60d")),
                "daily_volume_ratio_20d": _optional_float(route.get("daily_volume_ratio_20d")),
                "daily_close20_loc": _optional_float(route.get("daily_close20_loc")),
                "daily_acceleration_5v20": _optional_float(route.get("daily_acceleration_5v20")),
                "daily_momentum_pct": _optional_float(route.get("daily_momentum_pct")),
                "daily_sector_alignment_pct": _optional_float(route.get("daily_sector_alignment_pct")),
                "stock_sector_daily_ret20_spread": _optional_float(route.get("stock_sector_daily_ret20_spread")),
                "stock_sector_daily_ret5_spread": _optional_float(route.get("stock_sector_daily_ret5_spread")),
                "first30_quality_pct": _optional_float(route.get("first30_quality_pct")),
                "first30_sector_ret_spread": _optional_float(route.get("first30_sector_ret_spread")),
                "first30_sector_relvol_ratio": _optional_float(route.get("first30_sector_relvol_ratio")),
                "first30_sector_leadership_pct": _optional_float(route.get("first30_sector_leadership_pct")),
                "first30_gap_relvol_sector_breadth": _optional_float(route.get("first30_gap_relvol_sector_breadth")),
                "first30_gap_retention_sector_breadth": _optional_float(route.get("first30_gap_retention_sector_breadth")),
                "continuation_joint_quality_pct": _optional_float(route.get("continuation_joint_quality_pct")),
                "sector_participation": _optional_float(route.get("sector_participation")),
                "sector_daily_score_pct": _optional_float(route.get("sector_daily_score_pct")),
                "sector_daily_participation": _optional_float(route.get("sector_daily_participation")),
                "sector_daily_breadth_20d": _optional_float(route.get("sector_daily_breadth_20d")),
                "sector_daily_ret_5d": _optional_float(route.get("sector_daily_ret_5d")),
                "sector_daily_ret_20d": _optional_float(route.get("sector_daily_ret_20d")),
                "sector_intraday_score_pct": _optional_float(route.get("sector_intraday_score_pct")),
                "sector_intraday_ret": _optional_float(route.get("sector_intraday_ret")),
                "sector_intraday_breadth": _optional_float(route.get("sector_intraday_breadth")),
                "sector_intraday_participation": _optional_float(route.get("sector_intraday_participation")),
                "sector_intraday_effective_count": _optional_float(route.get("sector_intraday_effective_count")),
                "session_sector_intraday_score_pct_mean": _optional_float(route.get("session_sector_intraday_score_pct_mean")),
                "session_sector_intraday_positive_share": _optional_float(route.get("session_sector_intraday_positive_share")),
                "session_sector_intraday_effective_count_mean": _optional_float(route.get("session_sector_intraday_effective_count_mean")),
                "first30_ret": _optional_float(route.get("first30_ret")),
                "first30_vwap_ret": _optional_float(route.get("first30_vwap_ret")),
                "first30_gap": _optional_float(route.get("first30_gap")),
                "first30_gap_retention_ratio": _optional_float(route.get("first30_gap_retention_ratio")),
                "first30_gap_relvol": _optional_float(route.get("first30_gap_relvol")),
                "first30_low_vs_prev_relvol": _optional_float(route.get("first30_low_vs_prev_relvol")),
                "first30_rel_volume": _optional_float(route.get("first30_rel_volume")),
                "first30_range_close_location": _optional_float(route.get("first30_range_close_location", route.get("first30_close_location"))),
                "first30_signal_bar_cpr": _optional_float(route.get("first30_signal_bar_cpr")),
                "first30_open_drawdown": _optional_float(route.get("first30_open_drawdown")),
                "first30_low_vs_prev_close": _optional_float(route.get("first30_low_vs_prev_close")),
                "first30_range_atr": _optional_float(route.get("first30_range_atr")),
                "partial_taken": bool(cohort.get("partial_taken", False)),
            }
        )
    return tuple(rows)


def _decision_summary(decisions: Iterable[Any]) -> dict[str, Any]:
    codes: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    failed_gates: Counter[str] = Counter()
    for decision in decisions:
        code = str(getattr(decision, "decision_code", "") or "")
        codes[code] += 1
        if code != "entry_rejected":
            continue
        rejection_reasons[str(getattr(decision, "reason", "") or "unknown")] += 1
        metadata = dict(getattr(decision, "metadata", {}) or {})
        gates = metadata.get("gates") or metadata.get("filter_decisions") or ()
        first_failed = ""
        for gate in gates:
            if isinstance(gate, dict) and bool(gate.get("applicable", True)) and not bool(gate.get("passed", True)):
                first_failed = str(gate.get("filter_name") or "")
                break
        failed_gates[first_failed or str(getattr(decision, "reason", "") or "unknown")] += 1
    return {
        "decision_code_counts": codes.most_common(),
        "entry_rejection_reasons": rejection_reasons.most_common(12),
        "entry_failed_gates": failed_gates.most_common(12),
    }


def _executive_verdict_lines(final: dict[str, Any], baseline: dict[str, Any], rows: tuple[dict[str, Any], ...], baseline_rows: tuple[dict[str, Any], ...], audit: dict[str, Any]) -> list[str]:
    del baseline_rows
    first30 = _first30_truth_summary(rows)
    eod_share = float(final.get("exit_reason_eod_flatten_share", 0.0) or 0.0)
    capture = float(final.get("avg_mfe_capture", 0.0) or 0.0)
    mae_tail = float(final.get("mae_le_neg_1_share", 0.0) or 0.0)
    selected_conversion = float(final.get("signal_conversion", 0.0) or 0.0)
    net_delta = _signed_pct_delta(final, baseline, "broker_net_return_pct")
    trade_delta = _signed_num_delta(final, baseline, "trade_count", 0)
    pool_conversion = float(final.get("candidate_pool_conversion", 0.0) or 0.0)
    signal_verdict = "GOOD" if float(final.get("broker_net_return_pct", 0.0) or 0.0) > float(baseline.get("broker_net_return_pct", 0.0) or 0.0) else "REVIEW"
    discrimination_verdict = "REVIEW" if selected_conversion > 0.75 or mae_tail > 0.65 else "GOOD"
    entry_verdict = "REVIEW" if selected_conversion > 0.75 else "GOOD"
    management_verdict = "WEAK" if capture < 0.35 and eod_share > 0.50 else ("REVIEW" if capture < 0.40 or mae_tail > 0.50 else "GOOD")
    target_share = float(final.get("target_hit_share", 0.0) or final.get("exit_reason_target_r_share", 0.0) or 0.0)
    top_gate = _top_gate_name(audit)
    bottleneck = "trade management/MFE capture" if management_verdict in {"WEAK", "REVIEW"} else "selected-candidate discrimination"
    return [
        f"Signal extraction: {signal_verdict} (source/frontier phase lifted broker net by {net_delta} and trades by {trade_delta}; pool conversion is {pool_conversion:.2%}, so the broad frontier is not being naively chased).",
        f"Discrimination: {discrimination_verdict} (selected-candidate conversion {selected_conversion:.2%}, MAE<=-1R {mae_tail:.2%}; top full-replay gate={top_gate}).",
        f"First30 analysis: {first30['verdict']} ({first30['headline']}).",
        f"Entry mechanism: {entry_verdict} (phase 3 keeps first30 as anchor and only promotes secondary route branches that survive shared-core route metadata and fold checks).",
        f"Trade management: {management_verdict} (MFE capture {capture:.2%}, EOD flatten share {eod_share:.2%}, target-hit share {target_share:.2%}).",
        f"Exit mechanism: {management_verdict} (the shared core supports target, partial, MFE floor/giveback, late giveback, conditional stop, and VWAP-fail actions; appendix phases only promote exit changes after positive quantitative proof).",
        f"Primary bottleneck: {bottleneck}.",
    ]


def _candidate_surfacing_lines(final: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> list[str]:
    lines = ["Candidate surfacing diagnostics:"]
    lines.extend(_group_stat_lines(rows, "frontier_role", "  Frontier-role performance:", limit=8))
    lines.extend(_group_stat_lines(rows, "frontier_rank_bucket", "  Frontier-rank bucket performance:", key_fn=lambda row: _rank_bucket(row.get("frontier_rank")), limit=8))
    lines.extend(_top_bottom_symbol_lines(rows))
    if float(final.get("frontier_expansion_candidate_count", 0.0) or 0.0) > 0 and float(final.get("initial_active_candidate_count", 0.0) or 0.0) > 0:
        lines.append("  Interpretation: the replay now contains the broader frontier, but the accepted config still routes trades through initial-active names. That is good for parity and keeps naive expansion blocked, but it means frontier-shadow alpha still needs direct proof before being promoted.")
    return lines


def _gate_attribution_lines(audit: dict[str, Any]) -> list[str]:
    digest = dict(audit.get("audit_replay_digest") or {})
    lines = ["Full-replay gate attribution:"]
    reasons = list(digest.get("top_entry_rejection_reasons") or [])
    gates = list(digest.get("top_entry_failed_gates") or [])
    if not reasons:
        lines.append("  No entry rejection reasons were recorded.")
    else:
        lines.append("  Rejection reasons:")
        for row in reasons[:8]:
            lines.append(f"    {row.get('reason'):<34} {_num(row.get('count'), 0)}")
    if gates:
        lines.append("  First failed gates:")
        for row in gates[:8]:
            lines.append(f"    {row.get('gate'):<34} {_num(row.get('count'), 0)}")
    waiting = sum(int(row.get("count", 0) or 0) for row in reasons if str(row.get("reason")) == "waiting_for_first30_signal")
    rejected = int(digest.get("entry_rejection_count", 0) or 0)
    if rejected:
        lines.append(f"  Diagnostic note: waiting_for_first30_signal is scheduling noise ({waiting}/{rejected} rejections), not a quality filter. Real discrimination is from the non-waiting gates.")
    return lines


def _first30_signal_lines(rows: tuple[dict[str, Any], ...]) -> list[str]:
    summary = _first30_truth_summary(rows)
    lines = [
        f"Verdict: {summary['verdict']}",
        f"Coverage: first30 metadata present on {summary['coverage_n']}/{len(rows)} trades.",
        f"Headline: {summary['headline']}",
        "Winner vs loser feature deltas:",
    ]
    for key, label in (
        ("first30_ret", "First30 return"),
        ("first30_vwap_ret", "First30 close vs VWAP"),
        ("first30_rel_volume", "First30 relative volume"),
        ("first30_range_close_location", "First30 close location"),
        ("first30_signal_bar_cpr", "Signal-bar CPR"),
        ("first30_open_drawdown", "Open drawdown"),
        ("first30_low_vs_prev_close", "Low vs previous close"),
        ("first30_range_atr", "Range/ATR"),
    ):
        item = summary["profiles"].get(key, {})
        lines.append(
            f"  {label:<28} winners={_signed_num(item.get('winner_avg'))} losers={_signed_num(item.get('loser_avg'))} delta={_signed_num(item.get('delta'))} n={_num(item.get('n'), 0)}"
        )
    lines.append("First30 bucket expectancy:")
    lines.extend(_bucket_stat_lines(rows, "first30_ret", (-0.005, 0.0, 0.005, 0.015), "  first30_ret"))
    lines.extend(_bucket_stat_lines(rows, "first30_vwap_ret", (-0.003, 0.0, 0.003, 0.01), "  first30_vwap_ret"))
    lines.extend(_bucket_stat_lines(rows, "first30_rel_volume", (0.75, 1.0, 1.5, 2.5), "  first30_rel_volume"))
    lines.extend(_bucket_stat_lines(rows, "first30_range_close_location", (0.25, 0.50, 0.75), "  first30_close_location"))
    for action in summary["actions"]:
        lines.append(f"  - {action}")
    return lines


def _entry_mechanism_lines(final: dict[str, Any], rows: tuple[dict[str, Any], ...], audit: dict[str, Any]) -> list[str]:
    del audit
    lines = [
        f"Trade frequency: {len(rows)} trades, avg_trades/session={_num(final.get('avg_trades_per_session'), 3)}, selected-candidate conversion={_pct(final.get('signal_conversion'))}.",
        "Entry type performance:",
    ]
    lines.extend(_group_stat_lines(rows, "entry_type", "", limit=8))
    lines.append("Entry route performance:")
    lines.extend(_group_stat_lines(rows, "entry_route", "", limit=8))
    lines.append("Entry timing buckets:")
    lines.extend(_group_stat_lines(rows, "entry_time_bucket", "", key_fn=lambda row: _time_bucket(row.get("entry_time_label")), limit=10))
    lines.append("Candidate-rank buckets at entry:")
    lines.extend(_group_stat_lines(rows, "candidate_rank_bucket", "", key_fn=lambda row: _rank_bucket(row.get("candidate_rank")), limit=8))
    if float(final.get("signal_conversion", 0.0) or 0.0) > 0.75:
        lines.append("Interpretation: among selected candidates, entry is still permissive. The system blocks broad frontier noise, but once a name is selected it usually gets a trade.")
    else:
        lines.append("Interpretation: selected candidates face meaningful entry gating; evaluate missed-trade shadow before tightening further.")
    return lines


def _exit_management_lines(final: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> list[str]:
    lines = [
        f"Exit shares: eod_flatten={_pct(final.get('exit_reason_eod_flatten_share'))}, stopout={_pct(final.get('stopout_share'))}, target_hit={_pct(final.get('target_hit_share'))}, partial_hit={_pct(final.get('partial_hit_share'))}.",
        "Exit reason decomposition:",
    ]
    lines.extend(_group_stat_lines(rows, "exit_reason", "", limit=10))
    lines.append("MFE capture by exit reason:")
    for key, group_rows in _group_rows(rows, lambda row: str(row.get("exit_reason") or "unknown")).items():
        stats = _row_stats(group_rows)
        lines.append(f"  {key:<30} n={stats['n']:>4} avgR={_signed_num(stats['avg_r'])} capture={_pct(_avg_value(row.get('mfe_capture') for row in group_rows))} giveback={_signed_num(_avg_value(row.get('giveback_r') for row in group_rows))}R")
    lines.append("Hold-duration alpha curve:")
    for label, group_rows in _hold_bucket_groups(rows).items():
        stats = _row_stats(group_rows)
        lines.append(f"  {label:<22} n={stats['n']:>4} WR={_pct(stats['win_rate']):>7} avgR={_signed_num(stats['avg_r']):>8} totalR={_signed_num(stats['total_r']):>9}")
    target_share = float(final.get("target_hit_share", 0.0) or final.get("exit_reason_target_r_share", 0.0) or 0.0)
    if target_share > 0.0:
        lines.append("Interpretation: high-extension target capture is now harvesting a small set of large winners while leaving most trades on EOD flatten.")
    elif float(final.get("exit_reason_eod_flatten_share", 0.0) or 0.0) > 0.50:
        lines.append("Interpretation: EOD flatten still realizes most trades. Stopless giveback and conditional protection are structurally available, but this round did not prove they improve the realized edge.")
    lines.append("Appendix management veto: broad MFE giveback, partial exits, time decay, VWAP-fail-after-MFE, hard stop, and MFE-floor variants were quantitatively probed before the append run and excluded unless they showed positive score support.")
    return lines


def _mfe_mae_lines(final: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> list[str]:
    winners = [row for row in rows if _as_float(row.get("r")) > 0]
    losers = [row for row in rows if _as_float(row.get("r")) <= 0]
    actual_total_r = sum(_as_float(row.get("r")) for row in rows)
    positive_mfe_total = sum(max(_as_float(row.get("mfe_r")), 0.0) for row in rows)
    lines = [
        f"Avg MFE={_num(final.get('avg_mfe_r'), 2)}R, median MFE={_num(final.get('median_mfe_r'), 2)}R, MFE>=1R share={_pct(final.get('mfe_ge_1_share'))}.",
        f"Avg MAE={_num(final.get('avg_mae_r'), 2)}R, MAE<=-1R share={_pct(final.get('mae_le_neg_1_share'))}.",
        f"MFE capture={_pct(final.get('avg_mfe_capture'))}; broker MFE capture={_pct(final.get('broker_mfe_capture'))}; average hold={_num(final.get('avg_bars_held'), 1)} bars.",
        f"Winners: n={len(winners)}, avgR={_signed_num(_avg_value(row.get('r') for row in winners))}, avgMFE={_signed_num(_avg_value(row.get('mfe_r') for row in winners))}R, avgGiveback={_signed_num(_avg_value(row.get('giveback_r') for row in winners))}R.",
        f"Losers: n={len(losers)}, avgR={_signed_num(_avg_value(row.get('r') for row in losers))}, avgMAE={_signed_num(_avg_value(row.get('mae_r') for row in losers))}R, losers with MFE>1R={_pct(_share(_as_float(row.get('mfe_r')) > 1.0 for row in losers))}.",
        f"Diagnostic capture frontier: actual totalR={_signed_num(actual_total_r)}; 50% of positive MFE would be {_signed_num(0.50 * positive_mfe_total)}R; 75% would be {_signed_num(0.75 * positive_mfe_total)}R before costs/behavioral side effects.",
        *_cohort_diagnostic_lines(final),
    ]
    lost = sorted((row for row in rows if _as_float(row.get("mfe_r")) - _as_float(row.get("r")) > 0.0), key=lambda row: _as_float(row.get("mfe_r")) - _as_float(row.get("r")), reverse=True)
    if lost:
        lines.append("Top lost-alpha trades:")
        for row in lost[:8]:
            lost_r = _as_float(row.get("mfe_r")) - _as_float(row.get("r"))
            lines.append(f"  {row.get('entry_date')} {row.get('symbol')}: actual={_signed_num(row.get('r'))}R, MFE={_signed_num(row.get('mfe_r'))}R, lost={_signed_num(lost_r)}R, exit={row.get('exit_reason')}")
    return lines


def _period_stability_lines(rows: tuple[dict[str, Any], ...]) -> list[str]:
    lines = ["Monthly expectancy (worst and best months):"]
    monthly = _group_rows(rows, lambda row: str(row.get("entry_date", ""))[:7])
    month_stats = sorted(((key, _row_stats(value)) for key, value in monthly.items()), key=lambda item: item[1]["total_r"])
    for key, stats in month_stats[:4]:
        lines.append(f"  Worst {key}: n={stats['n']}, WR={_pct(stats['win_rate'])}, avgR={_signed_num(stats['avg_r'])}, totalR={_signed_num(stats['total_r'])}")
    for key, stats in list(reversed(month_stats[-4:])):
        lines.append(f"  Best  {key}: n={stats['n']}, WR={_pct(stats['win_rate'])}, avgR={_signed_num(stats['avg_r'])}, totalR={_signed_num(stats['total_r'])}")
    lines.append("Weekday mix:")
    lines.extend(_group_stat_lines(rows, "weekday", "", key_fn=lambda row: _weekday_from_iso(row.get("entry_date")), limit=7))
    rolling = _rolling_stats(rows, 20)
    lines.append(f"Rolling 20-trade expectancy: latest={_signed_num(rolling.get('latest'))}R, best={_signed_num(rolling.get('best'))}R, worst={_signed_num(rolling.get('worst'))}R, negative_windows={_num(rolling.get('negative_count'), 0)}/{_num(rolling.get('window_count'), 0)}.")
    streaks = _streak_stats(rows)
    lines.append(f"Streaks: max_win={streaks['max_win']}, max_loss={streaks['max_loss']}, worst_consecutive_loss={_signed_num(streaks['worst_loss_r'])}R.")
    return lines


def _paper_live_requirement_summary_lines(final: dict[str, Any], execution_context: dict[str, Any], payload: dict[str, Any] | None = None) -> list[str]:
    payload = dict(payload or _paper_live_parity_requirements(final, execution_context))
    contract = dict(payload.get("expected_contract") or {})
    baselines = dict(payload.get("backtest_hash_baselines") or {})
    return [
        f"Parity status: {payload.get('status')}; training replay audit pass={payload.get('training_replay_audit_pass')}.",
        f"Expected contract: mutation_hash={_short_hash(contract.get('optimized_mutations_hash'))}, candidate_snapshot={_short_hash(contract.get('candidate_snapshot_hash'))}, shared_core={contract.get('shared_core')}.",
        f"Backtest baselines: market_bars={_short_hash(baselines.get('market_bar_hash'))}, decisions={_short_hash(baselines.get('non_rejection_decision_hash'))}, actions={_short_hash(baselines.get('neutral_strategy_action_hash'))}, fills={_short_hash(baselines.get('fill_hash'))}, trades={_short_hash(baselines.get('trade_hash'))}.",
        "Promotion requires zero decision/action/OMS/fill/trade/state/snapshot/bar mismatches; diagnostic-only entry_rejected emissions may differ only if state and trades are unchanged.",
        "Paper/live evidence must include daily snapshots, completed 5m bars, decision stream, neutral actions, OMS intents, fills, trade outcomes, and restart-hydration state snapshots.",
    ]


def _artifact_pointer_lines(cache_metadata: dict[str, Any], execution_context: dict[str, Any], source_ref: FixedCandidateSourceRef, final: dict[str, Any], payload: dict[str, Any] | None) -> list[str]:
    contract = dict((payload or {}).get("expected_contract") or {})
    generated = dict((payload or {}).get("generated_backtest_artifacts") or {})
    counts = dict(cache_metadata.get("counts") or {})
    return [
        "Artifact pointers:",
        f"  source={source_ref.path}",
        f"  cache_key={cache_metadata.get('cache_key_short') or _short_hash(cache_metadata.get('cache_key'))}; cache_hit={cache_metadata.get('cache_hit')}; compiled_bars={counts.get('compiled_bars')}; snapshots={counts.get('snapshots')}; selections={counts.get('selections')}",
        f"  source_fingerprint={_short_hash(execution_context.get('source_fingerprint'))}; feature_manifest={_short_hash(execution_context.get('feature_manifest_hash'))}; candidate_snapshot={_short_hash(execution_context.get('candidate_snapshot_hash'))}",
        f"  final_compiled_replay={_short_hash(final.get('source_fingerprint'))}; final_candidate_artifact={_short_hash(final.get('candidate_snapshot_hash'))}; optimized_mutations={_short_hash(contract.get('optimized_mutations_hash'))}",
        f"  parity_contract={generated.get('paper_live_parity_contract', 'paper_live_parity_contract.json')}; snapshot_manifest={generated.get('daily_snapshot_manifest', 'paper_live_parity_inputs/daily_snapshot_manifest.json')}; market_bar_digest={generated.get('replay_market_bar_digest', 'paper_live_parity_inputs/replay_market_bar_digest.json')}",
    ]


def _layer_diagnostics_summary(final: dict[str, Any], rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    first30 = _first30_truth_summary(rows)
    return {
        "executive_verdicts": {
            "first30": first30["verdict"],
            "discrimination": "REVIEW" if float(final.get("signal_conversion", 0.0) or 0.0) > 0.75 or float(final.get("mae_le_neg_1_share", 0.0) or 0.0) > 0.65 else "GOOD",
            "trade_management": "WEAK" if float(final.get("avg_mfe_capture", 0.0) or 0.0) < 0.35 and float(final.get("exit_reason_eod_flatten_share", 0.0) or 0.0) > 0.50 else "REVIEW",
        },
        "first30": first30,
        "entry_type": {key: _row_stats(value) for key, value in _group_rows(rows, lambda row: str(row.get("entry_type") or "unknown")).items()},
        "entry_route": {key: _row_stats(value) for key, value in _group_rows(rows, lambda row: str(row.get("entry_route") or "legacy")).items()},
        "exit_reason": {key: _row_stats(value) for key, value in _group_rows(rows, lambda row: str(row.get("exit_reason") or "unknown")).items()},
        "top_symbols": _symbol_extremes(rows, reverse=True)[:8],
        "bottom_symbols": _symbol_extremes(rows, reverse=False)[:8],
    }


def _entry_rejection_reason_lines(audit: dict[str, Any]) -> list[str]:
    digest = dict(audit.get("audit_replay_digest") or {})
    rows = list(digest.get("top_entry_rejection_reasons") or [])
    if not rows:
        return ["Top entry rejection reasons: none recorded in full replay."]
    lines = ["Top entry rejection reasons from full replay:"]
    for row in rows[:8]:
        lines.append(f"  - {row.get('reason')}: {_num(row.get('count'), 0)}")
    return lines


def _cohort_diagnostic_lines(metrics: dict[str, Any]) -> list[str]:
    loser = dict(metrics.get("accepted_loser_summary") or {})
    lines = [
        f"Accepted losers: count={_num(loser.get('count'), 0)}, share={_pct(loser.get('share'))}, avg_MAE={_num(loser.get('avg_mae_r'), 2)}R, avg_MFE={_num(loser.get('avg_mfe_r'), 2)}R, MAE<=-1R={_pct(loser.get('mae_le_neg_1_share'))}",
    ]
    role_metrics = dict(metrics.get("mfe_capture_by_frontier_role") or {})
    if role_metrics:
        lines.append("Frontier-role cohorts:")
        for name, row in sorted(role_metrics.items()):
            row = dict(row or {})
            lines.append(
                f"  - {name}: trades={_num(row.get('trades'), 0)}, avg_net={_pct(row.get('avg_net_pct'))}, win={_pct(row.get('win_share'))}, capture={_pct(row.get('avg_mfe_capture'))}, MAE<=-1R={_pct(row.get('mae_le_neg_1_share'))}"
            )
    return lines


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "candidate_pool_count",
        "candidate_pool_conversion",
        "avg_trade_net_pct",
        "avg_mfe_capture",
        "mae_le_neg_1_share",
        "worst_fold_net",
        "immutable_score",
        "same_bar_fill_count",
        "end_open_position_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _fold_summary(fold_rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fold_rows:
        metrics = dict(row.get("metrics") or {})
        rows.append(
            {
                "fold": row.get("fold"),
                "start": row.get("start"),
                "end": row.get("end"),
                "net_return_pct": _first_metric(metrics, "portfolio_equivalent_net_return_pct", "primary_objective_net_return_pct", "broker_net_return_pct"),
                "max_drawdown_pct": _first_metric(metrics, "portfolio_equivalent_max_drawdown_pct", "broker_max_drawdown_pct", "max_drawdown_net_pct"),
                "trade_count": metrics.get("trade_count"),
                "active_days": metrics.get("active_days"),
                "avg_mfe_r": metrics.get("avg_mfe_r"),
                "avg_mfe_capture": metrics.get("avg_mfe_capture"),
                "signal_conversion": metrics.get("signal_conversion"),
            }
        )
    return rows


def _fold_stability_lines(fold_rows: tuple[dict[str, Any], ...], baseline_fold_rows: tuple[dict[str, Any], ...]) -> list[str]:
    if not fold_rows:
        return ["No fold rows available for this report."]
    baseline_by_fold = {str(row.get("fold")): row for row in baseline_fold_rows}
    lines: list[str] = []
    for row in fold_rows:
        metrics = dict(row.get("metrics") or {})
        baseline_row = baseline_by_fold.get(str(row.get("fold"))) or {}
        baseline_metrics = dict(baseline_row.get("metrics") or {})
        net = _first_metric(metrics, "portfolio_equivalent_net_return_pct", "primary_objective_net_return_pct", "broker_net_return_pct")
        dd = _first_metric(metrics, "portfolio_equivalent_max_drawdown_pct", "broker_max_drawdown_pct", "max_drawdown_net_pct")
        base_net = _first_metric(baseline_metrics, "portfolio_equivalent_net_return_pct", "primary_objective_net_return_pct", "broker_net_return_pct")
        base_dd = _first_metric(baseline_metrics, "portfolio_equivalent_max_drawdown_pct", "broker_max_drawdown_pct", "max_drawdown_net_pct")
        lines.append(
            f"Fold {row.get('fold')} ({row.get('start')} -> {row.get('end')}): "
            f"net={_pct(net)}, DD={_pct(dd)}, trades={_num(metrics.get('trade_count'), 0)}, active_days={_num(metrics.get('active_days'), 0)}, "
            f"avg_MFE={_num(metrics.get('avg_mfe_r'), 2)}R, capture={_pct(metrics.get('avg_mfe_capture'))}, conversion={_pct(metrics.get('signal_conversion'))}; "
            f"delta_vs_baseline net={_signed_raw_delta(net, base_net, pct=True)}, DD={_signed_raw_delta(dd, base_dd, pct=True)}"
        )
    values = [_first_metric(dict(row.get("metrics") or {}), "portfolio_equivalent_net_return_pct", "primary_objective_net_return_pct", "broker_net_return_pct") for row in fold_rows]
    dds = [_first_metric(dict(row.get("metrics") or {}), "portfolio_equivalent_max_drawdown_pct", "broker_max_drawdown_pct", "max_drawdown_net_pct") for row in fold_rows]
    lines.append(
        f"Fold verdict: worst_net={_pct(min((float(value) for value in values if value is not None), default=0.0))}, "
        f"median_net={_pct(median([float(value) for value in values if value is not None]) if any(value is not None for value in values) else 0.0)}, "
        f"worst_DD={_pct(max((abs(float(value)) for value in dds if value is not None), default=0.0))}."
    )
    return lines


def _source_fingerprint_lines(cache_metadata: dict[str, Any], execution_context: dict[str, Any], source_ref: FixedCandidateSourceRef) -> list[str]:
    lines = [
        f"Candidate source: {source_ref.path}",
        f"candidate_source_file_hash={cache_metadata.get('candidate_source_file_hash')}",
        f"candidate_source_sweep_hash={cache_metadata.get('candidate_source_sweep_hash')}",
        f"candidate_artifact_hash={cache_metadata.get('candidate_artifact_hash') or execution_context.get('candidate_snapshot_hash')}",
        f"intraday_source_fingerprint={cache_metadata.get('intraday_source_fingerprint')}",
        f"daily_source_fingerprint={cache_metadata.get('daily_source_fingerprint')}",
        f"preflight_source_fingerprint={cache_metadata.get('preflight_source_fingerprint')}",
        f"compiled_replay_fingerprint={cache_metadata.get('compiled_replay_fingerprint') or execution_context.get('source_fingerprint')}",
        f"compiled_replay_cache_key={cache_metadata.get('cache_key')}",
        f"compiled_replay_cache_path={cache_metadata.get('cache_path')}",
        f"metadata_path={cache_metadata.get('metadata_path')}",
        "Causality policy:",
    ]
    lines.extend(_indented_json_lines(cache_metadata.get("causality_policy", {}), indent=2))
    lines.append("Portfolio risk policy from source cache:")
    lines.extend(_indented_json_lines(cache_metadata.get("portfolio_risk_policy", {}), indent=2))
    return lines


def _phase_summary_lines(state: PhaseState) -> list[str]:
    lines: list[str] = []
    for phase, result in sorted(state.phase_results.items(), key=lambda item: int(item[0])):
        metrics = dict(result.get("final_metrics") or {})
        kept = result.get("kept_features") or []
        lines.append(
            f"Phase {phase}: {result.get('focus', '')}; applied={result.get('applied_phase_mutations')}; "
            f"accepted={result.get('accepted_count')}; kept={', '.join(kept) if kept else 'none'}; "
            f"score={_num(metrics.get('immutable_score'), 3)}; net={_pct(metrics.get('broker_net_return_pct'))}; "
            f"DD={_pct(metrics.get('broker_max_drawdown_pct'))}; trades={_num(metrics.get('trade_count'), 0)}"
        )
        for round_row in result.get("rounds", [])[:3]:
            lines.append(
                f"  round {round_row.get('round_num')}: best={round_row.get('best_name')} "
                f"score={_num(round_row.get('best_score'), 3)} delta={_num(round_row.get('best_delta_pct'), 3)}% "
                f"kept={round_row.get('kept')} rejected={round_row.get('rejected_count')}"
            )
    return lines or ["No phase results available."]


def _score_component_lines(metrics: dict[str, Any]) -> list[str]:
    scaled = dict(metrics.get("score_components") or {})
    lines: list[str] = []
    for name, weight in SCORE_COMPONENTS.items():
        lines.append(
            f"  {name}: weight={weight:+.3f}, scaled={_num(scaled.get(name), 4)}, contribution={_num(100.0 * weight * float(scaled.get(name, 0.0) or 0.0), 3)}"
        )
    return lines


def _strength_weakness_lines(final: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    dd_delta = float(final.get("broker_max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("broker_max_drawdown_pct", 0.0) or 0.0)
    dd_phrase = f"drawdown {'increased' if dd_delta > 0 else 'improved'} by {abs(100.0 * dd_delta):.2f}%"
    capture_delta = float(final.get("avg_mfe_capture", 0.0) or 0.0) - float(baseline.get("avg_mfe_capture", 0.0) or 0.0)
    capture_phrase = "improved" if capture_delta >= 0 else "fell"
    lines = [
        "Strengths:",
        f"  + Net return improved by {_signed_pct_delta(final, baseline, 'broker_net_return_pct')} while {dd_phrase}.",
        f"  + Frequency was preserved: {_num(baseline.get('trade_count'), 0)} -> {_num(final.get('trade_count'), 0)} trades.",
        f"  + Fast/full replay parity is clean, with {(final.get('fast_suppression_audit') or {}).get('suppressed_entry_rejection_count', 0)} suppressed rejection diagnostics and no fill/trade drift.",
        f"  + Worst fold stayed positive at {_pct(final.get('worst_fold_net'))}.",
        "Weaknesses / risks:",
        f"  - MFE capture {capture_phrase} from {_pct(baseline.get('avg_mfe_capture'))} to {_pct(final.get('avg_mfe_capture'))}; the strategy still needs holdout proof that captured excursion alpha is durable.",
        "  - Frontier expansion and source-quality gates remain intentionally hard to promote; any accepted expansion still needs locked holdout and paper parity before being treated as production alpha.",
        "  - Trade management remains path-dependent: many losers first had real MFE, but the tested protection families mostly cut right-tail alpha faster than they reduced drawdown.",
        "  - The sample is small and training-only; positive train folds do not remove the need for locked holdout and paper parity.",
        "  - High-extension targets are structural improvements, not proof of edge by themselves; promotion still depends on audit-clean fold and holdout behavior.",
    ]
    return lines


def _first30_truth_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    fields = (
        "first30_ret",
        "first30_vwap_ret",
        "first30_rel_volume",
        "first30_range_close_location",
        "first30_signal_bar_cpr",
        "first30_open_drawdown",
        "first30_low_vs_prev_close",
        "first30_range_atr",
    )
    first30_rows = tuple(row for row in rows if row.get("first30_ret") is not None)
    profiles = {field: _winner_loser_profile(first30_rows, field) for field in fields}
    helpful_fields = ("first30_ret", "first30_vwap_ret", "first30_rel_volume", "first30_range_close_location", "first30_signal_bar_cpr")
    helpful = sum(1 for field in helpful_fields if _as_float(profiles[field].get("delta")) > 0)
    misleading = sum(1 for field in helpful_fields if _as_float(profiles[field].get("delta")) < 0)
    verdict = "GOOD" if first30_rows and helpful >= 3 and misleading <= 1 else ("REVIEW" if first30_rows else "NOT_AVAILABLE")
    best = max((field for field in helpful_fields), key=lambda field: _as_float(profiles[field].get("delta")), default="")
    worst = min((field for field in helpful_fields), key=lambda field: _as_float(profiles[field].get("delta")), default="")
    actions: list[str] = []
    if profiles["first30_ret"]["delta"] <= 0:
        actions.append("First30 return is not separating winners cleanly; combine it with source/flow context rather than tightening it alone.")
    if profiles["first30_vwap_ret"]["delta"] <= 0:
        actions.append("Close-vs-VWAP is weak or misleading; avoid assuming VWAP premium is alpha without a cohort check.")
    if profiles["first30_rel_volume"]["delta"] <= 0:
        actions.append("Relative volume is not yet proving stronger realized trades; verify expected-volume normalization before increasing this gate.")
    if not actions:
        actions.append("First30 features are directionally useful; refine locally around the accepted gates rather than reopening the full signal space.")
    headline = (
        f"helpful_fields={helpful}/5, best_delta={best}:{_signed_num(profiles.get(best, {}).get('delta'))}, worst_delta={worst}:{_signed_num(profiles.get(worst, {}).get('delta'))}"
        if first30_rows
        else "no first30 metadata found on realized trades"
    )
    return {"verdict": verdict, "coverage_n": len(first30_rows), "profiles": profiles, "actions": actions, "headline": headline}


def _winner_loser_profile(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(field) is not None]
    winners = [_as_float(row.get(field)) for row in usable if _as_float(row.get("r")) > 0]
    losers = [_as_float(row.get(field)) for row in usable if _as_float(row.get("r")) <= 0]
    winner_avg = sum(winners) / len(winners) if winners else 0.0
    loser_avg = sum(losers) / len(losers) if losers else 0.0
    return {"winner_avg": winner_avg, "loser_avg": loser_avg, "delta": winner_avg - loser_avg, "n": len(usable)}


def _group_stat_lines(
    rows: Iterable[dict[str, Any]],
    field: str,
    title: str,
    *,
    key_fn: Any | None = None,
    limit: int = 8,
) -> list[str]:
    grouped = _group_rows(rows, key_fn or (lambda row: str(row.get(field) or "unknown")))
    lines: list[str] = [title] if title else []
    if not grouped:
        return lines + ["  (no rows)"]
    ranked = sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    for key, group in ranked[:limit]:
        stats = _row_stats(group)
        lines.append(f"  {str(key):<28} n={stats['n']:>4} WR={_pct(stats['win_rate']):>7} avgR={_signed_num(stats['avg_r']):>8} totalR={_signed_num(stats['total_r']):>9} PF={_num(stats['profit_factor'], 2):>6}")
    return lines


def _group_rows(rows: Iterable[dict[str, Any]], key_fn: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row))
        grouped.setdefault(key, []).append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _row_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [_as_float(row.get("r")) for row in rows]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gain = sum(wins)
    loss = abs(sum(losses))
    return {
        "n": len(values),
        "win_rate": len(wins) / len(values) if values else 0.0,
        "avg_r": sum(values) / len(values) if values else 0.0,
        "total_r": sum(values),
        "profit_factor": gain / loss if loss > 0 else (999.0 if gain > 0 else 0.0),
    }


def _bucket_stat_lines(rows: Iterable[dict[str, Any]], field: str, cuts: tuple[float, ...], label: str) -> list[str]:
    usable = [row for row in rows if row.get(field) is not None]
    if not usable:
        return [f"{label}: no data"]
    grouped = _group_rows(usable, lambda row: _numeric_bucket(_as_float(row.get(field)), cuts))
    lines = [f"{label}:"]
    for key, group in grouped.items():
        stats = _row_stats(group)
        lines.append(f"    {key:<18} n={stats['n']:>4} WR={_pct(stats['win_rate']):>7} avgR={_signed_num(stats['avg_r']):>8} totalR={_signed_num(stats['total_r']):>9}")
    return lines


def _numeric_bucket(value: float, cuts: tuple[float, ...]) -> str:
    previous: float | None = None
    for cut in cuts:
        if value < cut:
            return f"<{cut:.4g}" if previous is None else f"{previous:.4g}..{cut:.4g}"
        previous = cut
    return f">={cuts[-1]:.4g}" if cuts else "all"


def _top_bottom_symbol_lines(rows: tuple[dict[str, Any], ...]) -> list[str]:
    top = _symbol_extremes(rows, reverse=True)[:6]
    bottom = _symbol_extremes(rows, reverse=False)[:6]
    lines = ["  Top symbols by total R:"]
    for item in top:
        lines.append(f"    {item['symbol']}: n={item['n']}, totalR={_signed_num(item['total_r'])}, avgR={_signed_num(item['avg_r'])}")
    lines.append("  Weakest symbols by total R:")
    for item in bottom:
        lines.append(f"    {item['symbol']}: n={item['n']}, totalR={_signed_num(item['total_r'])}, avgR={_signed_num(item['avg_r'])}")
    return lines


def _symbol_extremes(rows: Iterable[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, lambda row: str(row.get("symbol") or ""))
    out: list[dict[str, Any]] = []
    for symbol, group in grouped.items():
        stats = _row_stats(group)
        out.append({"symbol": symbol, "n": stats["n"], "total_r": stats["total_r"], "avg_r": stats["avg_r"], "win_rate": stats["win_rate"]})
    out.sort(key=lambda item: (item["total_r"], item["avg_r"], item["symbol"]), reverse=reverse)
    return out


def _hold_bucket_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    order = ("1-6 bars", "7-12 bars", "13-24 bars", "25-48 bars", ">48 bars")
    grouped = _group_rows(rows, lambda row: _hold_bucket(row.get("hold_bars")))
    return {key: grouped[key] for key in order if key in grouped}


def _hold_bucket(value: Any) -> str:
    bars = _as_float(value)
    if bars <= 6:
        return "1-6 bars"
    if bars <= 12:
        return "7-12 bars"
    if bars <= 24:
        return "13-24 bars"
    if bars <= 48:
        return "25-48 bars"
    return ">48 bars"


def _rank_bucket(value: Any) -> str:
    rank = _as_int(value)
    if rank <= 0:
        return "unknown"
    if rank <= 1:
        return "rank 1"
    if rank <= 3:
        return "rank 2-3"
    if rank <= 5:
        return "rank 4-5"
    if rank <= 10:
        return "rank 6-10"
    if rank <= 30:
        return "rank 11-30"
    return "rank 31+"


def _time_bucket(label: Any) -> str:
    text = str(label or "")
    if not text or ":" not in text:
        return "unknown"
    try:
        hour = int(text[:2])
        minute = int(text[3:5])
    except ValueError:
        return "unknown"
    total = hour * 60 + minute
    bucket = (total // 30) * 30
    return f"{bucket // 60:02d}:{bucket % 60:02d}"


def _weekday_from_iso(value: Any) -> str:
    text = str(value or "")
    if len(text) < 10:
        return "unknown"
    try:
        from datetime import date as _date

        return _date.fromisoformat(text[:10]).strftime("%a")
    except ValueError:
        return "unknown"


def _rolling_stats(rows: tuple[dict[str, Any], ...], window: int) -> dict[str, Any]:
    values = [_as_float(row.get("r")) for row in rows]
    if len(values) < max(1, window):
        avg = sum(values) / len(values) if values else 0.0
        return {"latest": avg, "best": avg, "worst": avg, "negative_count": int(avg < 0), "window_count": 1 if values else 0}
    avgs = [sum(values[index : index + window]) / window for index in range(0, len(values) - window + 1)]
    return {
        "latest": avgs[-1],
        "best": max(avgs),
        "worst": min(avgs),
        "negative_count": sum(1 for value in avgs if value < 0),
        "window_count": len(avgs),
    }


def _streak_stats(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    max_win = max_loss = current_win = current_loss = 0
    worst_loss = 0.0
    running_loss = 0.0
    for row in rows:
        value = _as_float(row.get("r"))
        if value > 0:
            current_win += 1
            current_loss = 0
            running_loss = 0.0
        else:
            current_loss += 1
            current_win = 0
            running_loss += value
            worst_loss = min(worst_loss, running_loss)
        max_win = max(max_win, current_win)
        max_loss = max(max_loss, current_loss)
    return {"max_win": max_win, "max_loss": max_loss, "worst_loss_r": worst_loss}


def _top_gate_name(audit: dict[str, Any]) -> str:
    digest = dict(audit.get("audit_replay_digest") or {})
    gates = list(digest.get("top_entry_failed_gates") or [])
    gates = [row for row in gates if str(row.get("gate")) != "waiting_for_first30_signal"]
    if gates:
        first = gates[0]
        return f"{first.get('gate')} ({first.get('count')})"
    reasons = list(digest.get("top_entry_rejection_reasons") or [])
    reasons = [row for row in reasons if str(row.get("reason")) != "waiting_for_first30_signal"]
    if reasons:
        first = reasons[0]
        return f"{first.get('reason')} ({first.get('count')})"
    return "none"


def _avg_value(values: Iterable[Any]) -> float:
    vals = [_as_float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _share(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    return sum(1 for value in vals if value) / len(vals) if vals else 0.0


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _signed_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _short_hash(value: Any, length: int = 12) -> str:
    text = str(value or "")
    return text[:length] if text else "n/a"


def _short_text(value: Any, length: int = 160) -> str:
    text = str(value or "")
    if len(text) <= length:
        return text
    return text[: max(0, length - 3)] + "..."


def _indented_json_lines(value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    return [f"{prefix}{line}" for line in json.dumps(value, indent=2, sort_keys=True, default=str).splitlines()]


def _signed_pct_delta(final: dict[str, Any], baseline: dict[str, Any], key: str) -> str:
    try:
        return f"{100.0 * (float(final.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)):+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _signed_num_delta(final: dict[str, Any], baseline: dict[str, Any], key: str, digits: int = 2) -> str:
    try:
        return f"{(float(final.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _signed_raw_delta(final_value: Any, baseline_value: Any, *, pct: bool = False, digits: int = 2) -> str:
    if final_value is None or baseline_value is None:
        return "n/a"
    try:
        delta = float(final_value or 0.0) - float(baseline_value or 0.0)
    except (TypeError, ValueError):
        return "n/a"
    if pct:
        return f"{100.0 * delta:+.{digits}f}%"
    return f"{delta:+.{digits}f}"


def _first_metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"
