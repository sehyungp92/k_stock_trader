from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from backtests.auto.shared.cache_keys import build_cache_key, stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.plugin_utils import mutation_signature
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, ScoredCandidate
from backtests.strategies.common.plugin_base import SharedStrategyPluginMixin, attach_official_metric_contract, build_execution_contract
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION
from strategy_olr.execution import OLRAllocationPlan, OLREntryPlan, OLRExitPlan, OLRTradeOutcome, simulate_olr_trade, summarize_olr_portfolio_proxy
from strategy_olr.research import afternoon_selection_from_contexts

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import IMMUTABLE_SCORE_COMPONENTS, PHASE_HARD_REJECTS, ULTIMATE_TARGETS, gate_criteria, olr_reject_reason, score_olr_phase
from .replay_cache import load_olr_real_replay_bundle
from .research_sweep import (
    DEFAULT_EXPECTED_UNIVERSE_SIZE,
    DEFAULT_HOLDOUT_DAYS,
    OLRResearchSweepDataset,
    _base_mutations,
    _training_config,
    afternoon_contexts_for_snapshots,
    prepare_research_sweep_dataset,
    research_snapshots_for_dataset,
    snapshots_for_experiment,
)
from .runner import StrategyBacktestResult, attach_overnight_labels_to_snapshots, compile_olr_replay_bundle, run_olr_backtest, snapshots_from_bundle


_OLR_PROXY_EVAL_VERSION = "olr-phase-proxy-v4-targeted-selected-recovery-scope"
_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS = 2
_OLR_HEARTBEAT_SECONDS = 30.0
_OLR_PER_CANDIDATE_TIMEOUT_SECONDS = 300.0
_OLR_REBUILD_TIMEOUT_SECONDS = 600.0
_OLR_MINIMUM_TIMEOUT_SECONDS = 420.0
_OLR_REBUILD_MINIMUM_TIMEOUT_SECONDS = 720.0
_OLR_MAX_EVAL_BATCH_SIZE = 2
_OLR_SELECTION_REBUILD_PHASES = {1, 2, 6}
_OLR_MIN_RELATIVE_MTM_RETENTION = 0.97
_OLR_MIN_FINAL_RELATIVE_MTM_RETENTION = 1.00
_OLR_MIN_RELATIVE_ENTRY_RETENTION = 0.90
_STAGE1_PREFIXES = (
    "olr.universe.",
    "olr.frontier.",
    "olr.discovery.",
    "olr.premarket.",
    "olr.research.",
    "olr.signal.",
)
_SELECTION_PREFIXES = (*_STAGE1_PREFIXES, "olr.afternoon.", "olr.overnight.")
_EXECUTION_PREFIXES = (*_SELECTION_PREFIXES, "olr.trade_plan.", "olr.execution.", "olr.cost.", "olr.robustness.")
_ALLOCATION_PREFIXES = (*_EXECUTION_PREFIXES, "olr.allocation.")


class OLROptimizationPlugin(SharedStrategyPluginMixin):
    """Shared phased-auto plugin for OLR direct official training replay."""

    name = "olr"
    num_phases = 6
    requires_full_diagnostics = False
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = dict(BASE_MUTATIONS)
    default_scoring_weights = dict(IMMUTABLE_SCORE_COMPONENTS)
    prefer_thread_evaluator_on_windows = False
    reject_evaluation_timeouts = True

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        output_dir: Path | None = None,
        max_workers: int | None = 2,
        capability_level: str = "real_replay",
    ):
        self.config = dict(config or {})
        self.config.setdefault("capability_level", capability_level)
        self.config.setdefault("live_parity_fill_timing", "completed_5m_signal_next_bar_or_resting_close_auction")
        self.config.setdefault("auction_mode", "resting_close_auction_after_14_30_decision")
        self.config.setdefault("artifact_promotion_policy", "training_only_until_holdout_and_paper_parity")
        self.config.setdefault("paper_live_parity_required", True)
        self.config.setdefault("official_performance", False)
        self.config.setdefault("promotion_status", "training_only_paper_live_pending")
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_workers = max(1, min(int(max_workers or 2), 2))
        self.capability_level = str(self.config.get("capability_level", capability_level))
        self.holdout_days = int(self.config.get("holdout_days", DEFAULT_HOLDOUT_DAYS) or DEFAULT_HOLDOUT_DAYS)
        self.initial_mutations = _merged_initial_mutations(self.config)
        self.initial_mutations_override = dict(self.initial_mutations)
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._result_cache: dict[str, StrategyBacktestResult] = {}
        self._bundle_cache: dict[str, tuple[Any, dict[date, Any]]] = {}
        self._proxy_metrics_cache: dict[str, dict[str, Any]] = {}
        self._stage1_cache: dict[str, dict[date, Any]] = {}
        self._afternoon_context_cache: dict[str, dict[date, dict[str, Any]]] = {}
        self._selection_cache: dict[str, dict[date, Any]] = {}
        self._outcome_cache: dict[str, tuple[OLRTradeOutcome, ...]] = {}
        self._cache_lock = threading.Lock()
        self.cache_dir = (self.output_dir / "olr_phase_eval_cache") if self.output_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.capability_level.lower() in {"synthetic", "dry_run"}:
            baseline = run_olr_backtest(self.config, self.initial_mutations)
            self.training_config = dict(self.config)
            self.dataset = None
            self.research_snapshots = {}
            self.eligible_snapshot_dates = ()
            self.train_bars = ()
            self.source_fingerprint = baseline.source_fingerprint
            self.feature_manifest_hash = baseline.feature_bundle_hash
            self.execution_context = self._build_execution_context(dict(baseline.metrics))
            self._remember_backtest_result(self.initial_mutations, baseline)
        else:
            self.training_config = _training_config(dict(self.config), self.holdout_days)
            self.dataset = None
            self.research_snapshots = {}
            self.eligible_snapshot_dates = ()
            self.train_bars = ()
            baseline = self._evaluate_result(self.initial_mutations)
            self.source_fingerprint = baseline.source_fingerprint
            self.feature_manifest_hash = baseline.feature_bundle_hash
            self.execution_context = self._build_execution_context(dict(baseline.metrics))

    def canonicalize_mutations(self, mutations: dict[str, Any] | None) -> dict[str, Any]:
        return _canonicalize_olr_mutations(mutations or {})

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        focus, focus_metrics = PHASE_FOCUS[phase]
        return PhaseSpec(
            focus=focus,
            candidates=get_phase_candidates(phase),
            gate_criteria_fn=lambda metrics, current_phase=phase: gate_criteria(current_phase, metrics, PHASE_HARD_REJECTS[current_phase]),
            scoring_weights=None,
            hard_rejects=dict(PHASE_HARD_REJECTS.get(phase, {})),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.001,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=None,
            ),
            max_rounds=1,
            prune_threshold=0.15,
            reject_streak_limit=2,
            phase_metric_basis="direct_official_training_replay_holdout_excluded",
            primary_promotion_metric="official_mtm_net_return_pct",
            official_metric_keys=("official_mtm_net_return_pct", "official_mtm_max_drawdown_pct", "official_mtm_sharpe"),
            promotion_requires_audit_pass=True,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        from .worker import init_worker, score_candidate

        cumulative_mutations = self.canonicalize_mutations(cumulative_mutations)
        base_result = self._evaluate_result(cumulative_mutations)
        reject = olr_reject_reason(phase, base_result.metrics, hard_rejects)
        baseline_score = score_olr_phase(phase, base_result.metrics, scoring_weights)
        baseline = ScoredCandidate("__baseline__", baseline_score, bool(reject), reject, base_result.metrics)
        rebuild_phase = phase in _OLR_SELECTION_REBUILD_PHASES
        if rebuild_phase and self.capability_level.lower() not in {"synthetic", "dry_run"}:
            from .replay_cache import warm_olr_stage1_replay_cache

            warm_olr_stage1_replay_cache(self.training_config, cumulative_mutations)
        return self._wrap_cached_evaluator(
            phase=phase,
            cumulative_mutations=cumulative_mutations,
            scoring_weights=scoring_weights,
            hard_rejects=hard_rejects,
            init_worker=init_worker,
            score_candidate=score_candidate,
            initargs=(self.training_config, phase, hard_rejects, scoring_weights, cumulative_mutations),
            heartbeat_seconds=_OLR_HEARTBEAT_SECONDS,
            per_candidate_timeout_seconds=(
                _OLR_REBUILD_TIMEOUT_SECONDS
                if rebuild_phase
                else _OLR_PER_CANDIDATE_TIMEOUT_SECONDS
            ),
            minimum_timeout_seconds=(
                _OLR_REBUILD_MINIMUM_TIMEOUT_SECONDS
                if rebuild_phase
                else _OLR_MINIMUM_TIMEOUT_SECONDS
            ),
            max_eval_batch_size=_OLR_MAX_EVAL_BATCH_SIZE,
            description=f"olr phase {phase}",
            baseline_result=baseline,
            reject_on_timeout=True,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        mutations = self.canonicalize_mutations(mutations)
        result = self._evaluate_result(mutations)
        return attach_official_metric_contract(
            dict(result.metrics),
            requires_audit_pass=True,
            audit_status="direct_official_training_replay_paper_live_pending",
            official_replay_pass=True,
            audit_pass=True,
            execution_contract=build_execution_contract(
                self,
                result.metrics,
                extra={
                    "holdout_excluded": True,
                    "paper_live_parity_status": "required_before_promotion",
                    "candidate_generation_cutoffs": {
                        "daily": "row_date < trade_date",
                        "stage2_intraday": "timestamp < 14:30 KST",
                    },
                },
            ),
        )

    def phase_acceptance_criteria(
        self,
        *,
        phase: int,
        base_mutations: dict[str, Any],
        base_metrics: dict[str, Any],
        final_metrics: dict[str, Any],
        greedy_result: GreedyResult,
    ) -> list[GateCriterion]:
        del base_mutations, greedy_result
        if phase < 4:
            return []
        min_mtm_retention = _OLR_MIN_FINAL_RELATIVE_MTM_RETENTION if phase >= self.num_phases else _OLR_MIN_RELATIVE_MTM_RETENTION
        base_return = _metric_float(base_metrics, "official_mtm_net_return_pct", "net_return_pct")
        final_return = _metric_float(final_metrics, "official_mtm_net_return_pct", "net_return_pct")
        if base_return > 0.0:
            return_ratio = final_return / max(base_return, 1e-12)
            return_passed = return_ratio >= min_mtm_retention
            return_actual = return_ratio
            return_target = min_mtm_retention
        else:
            return_actual = final_return - base_return
            return_target = -0.01
            return_passed = return_actual >= return_target
        base_entries = _metric_float(base_metrics, "entry_fill_count", "entry_level_trade_count", "total_trades")
        final_entries = _metric_float(final_metrics, "entry_fill_count", "entry_level_trade_count", "total_trades")
        entry_ratio = final_entries / max(base_entries, 1.0)
        return [
            GateCriterion(
                "hard_relative_mtm_non_regression",
                return_target,
                return_actual,
                return_passed,
            ),
            GateCriterion(
                "hard_entry_frequency_retention",
                _OLR_MIN_RELATIVE_ENTRY_RETENTION,
                entry_ratio,
                entry_ratio >= _OLR_MIN_RELATIVE_ENTRY_RETENTION,
            ),
        ]

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        del state
        return _format_olr_diagnostics(phase, metrics, greedy_result, self.execution_context)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result) + "\nEnhanced checks: replay snapshots are rebuilt from training-only source data; neutral actions route through OLR core and SimBroker only."

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final = self._evaluate_result(state.cumulative_mutations)
        text = _format_olr_diagnostics(self.num_phases, final.metrics, _greedy_from_state(state, final.metrics), self.execution_context)
        return EndOfRoundArtifacts(
            final_diagnostics_text=text,
            dimension_reports={
                "signal_extraction": "Stage 1 and Stage 2 are rebuilt through shared OLR research functions with holdout excluded; selector changes are judged by realized official replay plus label discrimination, not label proxy alone.",
                "signal_discrimination": "Score-band, top/bottom label spread, positive/negative label shares, and rejected-signal hygiene are promotion-visible metrics.",
                "entry_mechanism": "Entry route mutations use completed 5m bars and fill no earlier than a later bar, or a resting close-auction order submitted after the decision cutoff.",
                "trade_management": "Managed exits are neutral OLR actions handled by the same SimBroker path as baseline next-close exits.",
                "allocation": "Rank, score, and capped equal allocations remain shared-core sizing mutations with MTM drawdown as the risk basis.",
                "paper_live_parity": "Artifacts remain training-only until untouched holdout and paper/live order reconciliation validate the same shared-core action path.",
            },
            overall_verdict=(
                "OLR phased auto uses direct official training replay from the round-1 optimized baseline. "
                "The round is suitable for research promotion only; holdout and paper/live parity remain mandatory before production use."
            ),
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        del phase
        gaps = [
            "Holdout remains intentionally excluded from phased-auto search and must be rerun once a round candidate is selected.",
            "Paper/live parity still needs completed-bar action replay, order reconciliation, and auction fill/nonfill evidence before promotion.",
        ]
        if metrics.get("olr_discrimination_quality", 0.0) < 0.35:
            gaps.append("Signal rank still has weak monotonic realized-label separation; avoid trusting isolated score lift without score-band evidence.")
        if metrics.get("olr_alpha_capture", metrics.get("mfe_capture", 0.0)) < 0.20:
            gaps.append("MFE capture is still low; managed exits need cohort-level confirmation before relying on next-close drift alone.")
        return gaps

    def suggest_experiments(self, phase: int, metrics: dict[str, float], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del weaknesses, state
        if phase <= 2 and metrics.get("olr_score_top_loss_share", 0.0) > 0.55:
            return [
                Experiment(
                    "analysis_retry_stricter_top_band_rejection",
                    {
                        "olr.afternoon.score_calibration_mode": "exhaustion_adjusted",
                        "olr.afternoon.exhaustion_penalty": 25.0,
                        "olr.afternoon.max_exhaustion_score": 2.25,
                        "olr.afternoon.require_close_above_prev": True,
                    },
                )
            ]
        if phase in {3, 4} and metrics.get("olr_alpha_capture", metrics.get("mfe_capture", 0.0)) < 0.18:
            return [
                Experiment(
                    "analysis_retry_partial_be_target",
                    {
                        "olr.trade_plan.exit": {
                            "name": "managed_partial050_be_target150",
                            "mode": "managed",
                            "stop_mode": "decision_low",
                            "hard_stop_enabled": True,
                            "partial_trigger_r": 0.50,
                            "partial_fraction": 0.50,
                            "partial_stop_r": 0.0,
                            "target_r": 1.50,
                        }
                    },
                )
            ]
        return []

    def _score_candidate(
        self,
        name: str,
        candidate_mutations: dict[str, Any],
        current_mutations: dict[str, Any],
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ) -> ScoredCandidate:
        merged = self.canonicalize_mutations({**dict(current_mutations or {}), **dict(candidate_mutations or {})})
        metrics = self._evaluate_proxy_metrics(merged)
        score = score_olr_phase(phase, metrics, scoring_weights)
        reject = olr_reject_reason(phase, metrics, hard_rejects)
        return ScoredCandidate(name, 0.0 if reject else score, bool(reject), reject, metrics)

    def _evaluate_proxy_metrics(self, mutations: dict[str, Any]) -> dict[str, Any]:
        mutations = self.canonicalize_mutations(mutations)
        if self.capability_level.lower() in {"synthetic", "dry_run"}:
            result = run_olr_backtest(self.config, mutations)
            metrics = dict(result.metrics)
            metrics["phase_candidate_metric_basis"] = "synthetic_direct_replay"
            return metrics
        cache_key = self._proxy_cache_key(mutations)
        with self._cache_lock:
            cached = self._proxy_metrics_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        disk_payload = self._read_metric_cache("proxy", cache_key)
        if disk_payload:
            with self._cache_lock:
                self._proxy_metrics_cache[cache_key] = dict(disk_payload)
            return dict(disk_payload)
        snapshots = self._selection_snapshots(mutations)
        cfg = OLRConfig.from_mapping(self.dataset.config if self.dataset is not None else self.training_config, mutations)
        outcomes = self._proxy_outcomes(mutations, snapshots, cfg)
        selection_counts = _selection_counts(snapshots, cfg)
        allocation = _allocation_plan_from_config(cfg)
        portfolio = summarize_olr_portfolio_proxy(
            outcomes,
            session_dates=self.eligible_snapshot_dates,
            selection_counts=selection_counts,
            slot_count=cfg.overnight_slot_count,
            allocation=allocation,
            initial_equity=float(self.training_config.get("initial_equity", self.config.get("initial_equity", 10_000_000.0)) or 10_000_000.0),
            config=cfg,
        )
        metrics = _proxy_metrics_from_outcomes(outcomes, portfolio)
        metrics["strategy_core_version"] = OLR_CORE_VERSION
        metrics["replay_mode"] = "olr_shared_execution_proxy"
        metrics["phase_candidate_metric_basis"] = "train_only_shared_research_execution_proxy"
        metrics["official_metric_basis"] = "shared_execution_proxy_for_candidate_ranking"
        metrics["official_performance"] = False
        metrics["official_replay_pass"] = False
        metrics["audit_pass"] = False
        metrics["audit_status"] = "proxy_candidate_ranking_official_replay_on_phase_promotion"
        metrics["capability_level"] = self.capability_level
        metrics["holdout_excluded"] = True
        metrics["paper_live_parity_required"] = True
        metrics["paper_live_parity_status"] = "required_before_promotion"
        metrics["phase_score_component_count"] = float(len(IMMUTABLE_SCORE_COMPONENTS))
        metrics["source_fingerprint"] = self.dataset.source_fingerprint if self.dataset is not None else ""
        metrics["candidate_snapshot_hash"] = _snapshot_hash(snapshots)
        metrics["feature_manifest_hash"] = stable_signature(
            {
                "dataset": metrics["source_fingerprint"],
                "selection": metrics["candidate_snapshot_hash"],
                "proxy_version": _OLR_PROXY_EVAL_VERSION,
            }
        )
        metrics["selected_candidate_count"] = float(sum(selection_counts.values()))
        metrics["selected_day_count"] = float(sum(1 for count in selection_counts.values() if count > 0))
        metrics["avg_selected_per_day"] = float(mean(selection_counts.values())) if selection_counts else 0.0
        _augment_snapshot_label_metrics(metrics, snapshots)
        _augment_proxy_trade_alpha_metrics(metrics, outcomes)
        with self._cache_lock:
            self._proxy_metrics_cache[cache_key] = dict(metrics)
        self._write_metric_cache("proxy", cache_key, metrics)
        return dict(metrics)

    def _evaluate_result(self, mutations: dict[str, Any]) -> StrategyBacktestResult:
        mutations = self.canonicalize_mutations(mutations)
        key = mutation_signature(mutations)
        with self._cache_lock:
            cached = self._result_cache.get(key)
        if cached is not None:
            self._remember_backtest_result(mutations, cached)
            return cached
        if self.capability_level.lower() in {"synthetic", "dry_run"}:
            result = run_olr_backtest(self.config, mutations)
            snapshots = {}
        else:
            replay_bundle = load_olr_real_replay_bundle(self.training_config, mutations)
            snapshots = snapshots_from_bundle(replay_bundle)
            result = run_olr_backtest(self.training_config, mutations, replay_bundle=replay_bundle)
        self._augment_olr_metrics(result, snapshots, mutations)
        with self._cache_lock:
            self._result_cache[key] = result
        self._remember_backtest_result(mutations, result)
        return result

    def _build_replay_bundle(self, mutations: dict[str, Any]):
        key = mutation_signature(mutations)
        with self._cache_lock:
            cached = self._bundle_cache.get(key)
        if cached is not None:
            return cached
        if self.dataset is None:
            raise RuntimeError("OLR real replay dataset is not initialized")
        cfg = OLRConfig.from_mapping(self.dataset.config, mutations)
        snapshots = self._selection_snapshots(mutations)
        replay_bars, bar_scope = _filtered_training_bars_for_snapshots(self.dataset, snapshots, cfg)
        source_fingerprint = stable_signature(
            {
                "dataset": self.dataset.source_fingerprint,
                "mutations": mutations,
                "snapshots": {day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()},
                "holdout_days": self.holdout_days,
                "holdout_excluded": True,
                "bar_scope": bar_scope,
            }
        )
        replay_bundle = compile_olr_replay_bundle(
            replay_bars,
            snapshots,
            source_fingerprint=source_fingerprint,
            data_root=self.dataset.data_root,
            config=self.training_config,
        )
        replay_bundle.metadata["olr_feature_bundle_hash"] = stable_signature(
            {
                "dataset": self.dataset.source_fingerprint,
                "research_snapshots": _research_snapshot_hashes(self.research_snapshots),
                "mutations": mutations,
            }
        )
        replay_bundle.metadata["holdout_excluded"] = True
        replay_bundle.metadata["training_window"] = {
            "start": self.dataset.train_start.isoformat(),
            "end": self.dataset.train_end.isoformat(),
            "eligible_snapshot_end": self.eligible_snapshot_dates[-1].isoformat() if self.eligible_snapshot_dates else "",
            "sessions": len(self.dataset.trading_dates),
            "eligible_sessions": len(self.eligible_snapshot_dates),
            "replayed_selected_symbol_bars": len(replay_bars),
            "bar_scope": bar_scope,
        }
        result = (replay_bundle, snapshots)
        with self._cache_lock:
            self._bundle_cache[key] = result
        return result

    def _selection_snapshots(self, mutations: dict[str, Any]) -> dict[date, Any]:
        key = _selection_key(mutations)
        with self._cache_lock:
            cached = self._selection_cache.get(key)
        if cached is not None:
            return cached
        if self.dataset is None:
            raise RuntimeError("OLR real replay dataset is not initialized")
        cfg = OLRConfig.from_mapping(self.dataset.config, mutations)
        stage1 = self._stage1_snapshots(mutations)
        contexts = self._afternoon_contexts(mutations, stage1, cfg)
        stage2 = {
            day: afternoon_selection_from_contexts(snapshot, contexts.get(day, {}), cfg)
            for day, snapshot in sorted(stage1.items())
            if day in self.eligible_snapshot_dates
        }
        snapshots = attach_overnight_labels_to_snapshots(stage2, self.dataset.overnight_labels_by_key)
        with self._cache_lock:
            self._selection_cache[key] = snapshots
        return snapshots

    def _stage1_snapshots(self, mutations: dict[str, Any]) -> dict[date, Any]:
        key = _stage1_key(mutations)
        with self._cache_lock:
            cached = self._stage1_cache.get(key)
        if cached is not None:
            return cached
        if self.dataset is None:
            raise RuntimeError("OLR real replay dataset is not initialized")
        stage1 = snapshots_for_experiment(self.dataset, mutations, research_snapshots=self.research_snapshots)
        with self._cache_lock:
            self._stage1_cache[key] = stage1
        return stage1

    def _afternoon_contexts(self, mutations: dict[str, Any], stage1: dict[date, Any], cfg: OLRConfig) -> dict[date, dict[str, Any]]:
        key = _stage1_key(mutations)
        with self._cache_lock:
            cached = self._afternoon_context_cache.get(key)
        if cached is not None:
            return cached
        if self.dataset is None:
            raise RuntimeError("OLR real replay dataset is not initialized")
        contexts = afternoon_contexts_for_snapshots(self.dataset, stage1, cfg)
        with self._cache_lock:
            self._afternoon_context_cache[key] = contexts
        return contexts

    def _proxy_outcomes(
        self,
        mutations: dict[str, Any],
        snapshots: dict[date, Any],
        cfg: OLRConfig,
    ) -> tuple[OLRTradeOutcome, ...]:
        key = _execution_key(mutations)
        with self._cache_lock:
            cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached
        if self.dataset is None:
            raise RuntimeError("OLR real replay dataset is not initialized")
        entry_plan = _entry_plan_from_config(cfg)
        exit_plan = _exit_plan_from_config(cfg)
        next_by_day = _next_session_map(self.dataset.trading_dates)
        outcomes: list[OLRTradeOutcome] = []
        for day in self.eligible_snapshot_dates:
            snapshot = snapshots.get(day)
            if snapshot is None:
                continue
            next_day = next_by_day.get(day)
            if next_day is None:
                continue
            for candidate in tuple(getattr(snapshot, "candidates", ()) or ())[: max(1, int(cfg.overnight_slot_count))]:
                symbol = str(getattr(candidate, "symbol", "")).zfill(6)
                entry_bars = self.dataset.bars_by_key.get((day, symbol), ())
                next_bars = self.dataset.bars_by_key.get((next_day, symbol), ())
                outcome = simulate_olr_trade(day, symbol, entry_bars, next_bars, candidate, entry_plan, exit_plan, cfg)
                if outcome is not None:
                    outcomes.append(outcome)
        result = tuple(outcomes)
        with self._cache_lock:
            self._outcome_cache[key] = result
        return result

    def _proxy_cache_key(self, mutations: dict[str, Any]) -> str:
        return build_cache_key(
            "olr.phase_proxy_metrics",
            source_fingerprint=self.dataset.source_fingerprint if self.dataset is not None else self.source_fingerprint if hasattr(self, "source_fingerprint") else "",
            mutations=mutations,
            extra={
                "version": _OLR_PROXY_EVAL_VERSION,
                "holdout_days": self.holdout_days,
                "score_components": IMMUTABLE_SCORE_COMPONENTS,
                "training_start": getattr(self.dataset, "train_start", "") if self.dataset is not None else "",
                "training_end": getattr(self.dataset, "train_end", "") if self.dataset is not None else "",
            },
        )

    def _read_metric_cache(self, namespace: str, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{namespace}_{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("cache_version") != _OLR_PROXY_EVAL_VERSION:
            return None
        metrics = payload.get("metrics")
        return dict(metrics) if isinstance(metrics, dict) else None

    def _write_metric_cache(self, namespace: str, key: str, metrics: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        from backtests.auto.shared.phase_state import _atomic_write_json

        _atomic_write_json(
            {
                "cache_version": _OLR_PROXY_EVAL_VERSION,
                "key": key,
                "namespace": namespace,
                "metrics": dict(metrics),
            },
            self.cache_dir / f"{namespace}_{key}.json",
        )

    def _augment_olr_metrics(self, result: StrategyBacktestResult, snapshots: dict[date, Any], mutations: dict[str, Any]) -> None:
        metrics = result.metrics
        metrics["source_fingerprint"] = result.source_fingerprint
        metrics["feature_manifest_hash"] = result.feature_bundle_hash
        metrics["capability_level"] = result.capability_level
        metrics["holdout_excluded"] = True
        metrics["paper_live_parity_required"] = True
        metrics["paper_live_parity_status"] = "required_before_promotion"
        metrics["official_performance"] = False
        metrics["promotion_status"] = "training_only_paper_live_pending"
        metrics["phase_score_component_count"] = float(len(IMMUTABLE_SCORE_COMPONENTS))
        metrics["phase_score_spec_hash"] = build_cache_key("olr.phase_score", extra=IMMUTABLE_SCORE_COMPONENTS)
        metrics["baseline_mutation_hash"] = stable_signature(self.initial_mutations)
        metrics["mutation_hash"] = stable_signature(mutations)
        _augment_snapshot_label_metrics(metrics, snapshots)
        _augment_trade_alpha_metrics(metrics, result.trades)

    def _build_execution_context(self, baseline: dict[str, Any]) -> dict[str, Any]:
        raw_metric_key = build_cache_key(
            "olr.raw_metrics",
            source_fingerprint=str(baseline.get("source_fingerprint") or ""),
            mutations=self.initial_mutations,
            extra={"training_config": self.training_config, "holdout_excluded": True},
        )
        return {
            "shared_decision_core": True,
            "strategy_core_version": OLR_CORE_VERSION,
            "source_fingerprint": str(baseline.get("source_fingerprint") or ""),
            "feature_manifest_hash": str(baseline.get("feature_manifest_hash") or ""),
            "candidate_snapshot_hash": str(baseline.get("candidate_snapshot_hash") or ""),
            "live_parity_fill_timing": self.config["live_parity_fill_timing"],
            "auction_mode": self.config["auction_mode"],
            "initial_equity": self.config.get("initial_equity", ""),
            "cost_policy": baseline.get("cost_policy") or _cost_policy(self.config),
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "score_metric_basis": "direct_official_training_replay_plus_training_label_discrimination",
            "artifact_promotion_policy": self.config["artifact_promotion_policy"],
            "paper_live_parity_required": True,
            "paper_live_parity_status": "required_before_promotion",
            "paper_live_parity_contract": {
                "shared_core_required": True,
                "single_broker_path_required": True,
                "completed_bar_policy_required": True,
                "same_bar_fill_prohibited": True,
                "mtm_risk_required": True,
                "paper_order_reconciliation_required": True,
                "auction_fill_nonfill_reconciliation_required": True,
            },
            "holdout_excluded": True,
            "holdout_days": self.holdout_days,
            "train_start": getattr(self.dataset, "train_start", ""),
            "train_end": getattr(self.dataset, "train_end", ""),
            "risk_basis": "mark_to_market",
            "account_scope": "krx_cash_long_only",
            "diagnostics_version": "olr-phased-auto-v1",
            "raw_metric_cache_key": raw_metric_key,
        }


class _OLRBatchEvaluator:
    def __init__(
        self,
        plugin: OLROptimizationPlugin,
        *,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        max_workers: int,
    ):
        self._plugin = plugin
        self._phase = int(phase)
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._max_workers = max(1, min(int(max_workers or 1), 2))
        self._progress_callback = None

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        if self._max_workers <= 1 or len(candidates) <= 1:
            out = [
                self._plugin._score_candidate(candidate.name, candidate.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects)
                for candidate in candidates
            ]
            self._emit({"event": "batch_complete", "completed": len(out), "total": len(candidates)})
            return out
        out: list[ScoredCandidate | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix=f"olr-phase-{self._phase}") as executor:
            futures = {
                executor.submit(
                    self._plugin._score_candidate,
                    candidate.name,
                    candidate.mutations,
                    current_mutations,
                    self._phase,
                    self._scoring_weights,
                    self._hard_rejects,
                ): index
                for index, candidate in enumerate(candidates)
            }
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                out[index] = future.result()
                completed += 1
                self._emit({"event": "candidate_complete", "completed": completed, "total": len(candidates)})
        return [item for item in out if item is not None]

    def close(self) -> None:
        return None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def _emit(self, payload: dict[str, Any]) -> None:
        if callable(self._progress_callback):
            self._progress_callback(payload)


def _augment_snapshot_label_metrics(metrics: dict[str, Any], snapshots: dict[date, Any]) -> None:
    labels: list[tuple[float, float, float]] = []
    per_day_counts: list[int] = []
    rule_selected_counts: dict[str, int] = {}
    for snapshot in snapshots.values():
        candidates = list(getattr(snapshot, "candidates", ()) or ())
        per_day_counts.append(len(candidates))
        for candidate in candidates:
            meta = dict(getattr(candidate, "metadata", {}) or {})
            rule_name = _score_band_rule(meta)
            rule_selected_counts[rule_name] = rule_selected_counts.get(rule_name, 0) + 1
            if "close_to_close_label_pct" not in meta:
                continue
            labels.append(
                (
                    float(getattr(candidate, "selection_score", 0.0) or 0.0),
                    _float(meta.get("close_to_close_label_pct")),
                    _float(meta.get("next_session_mfe_label_pct")),
                )
            )
    metrics["selected_candidate_count"] = float(sum(per_day_counts))
    metrics["selected_day_count"] = float(sum(1 for count in per_day_counts if count > 0))
    metrics["avg_selected_per_day"] = float(mean(per_day_counts)) if per_day_counts else 0.0
    metrics["selected_with_label_count"] = float(len(labels))
    returns = [item[1] for item in labels]
    mfes = [item[2] for item in labels]
    metrics["selected_avg_close_to_close_label_pct"] = float(mean(returns)) if returns else 0.0
    metrics["selected_avg_mfe_label_pct"] = float(mean(mfes)) if mfes else 0.0
    metrics["olr_selected_positive_label_share"] = _share(returns, lambda value: value > 0.0)
    metrics["olr_selected_negative_label_share"] = _share(returns, lambda value: value <= 0.0)
    metrics["selected_negative_label_share"] = metrics["olr_selected_negative_label_share"]
    metrics["score_band_rule_selected_counts"] = dict(sorted(rule_selected_counts.items()))
    metrics["dynamic_overlay_selected_count"] = float(
        sum(count for rule, count in rule_selected_counts.items() if _is_dynamic_overlay_rule(rule))
    )
    score_stats = _score_band_stats(labels)
    metrics.update(score_stats)
    spread_quality = _clip01(score_stats["olr_score_top_bottom_label_spread_pct"] / 0.025)
    positive_quality = _clip01((metrics["olr_selected_positive_label_share"] - 0.35) / 0.35)
    negative_quality = _clip01((0.72 - metrics["olr_selected_negative_label_share"]) / 0.37)
    monotonicity = _clip01(score_stats["olr_score_monotonicity"])
    metrics["olr_discrimination_quality"] = _clip01(
        0.30 * positive_quality
        + 0.25 * negative_quality
        + 0.25 * spread_quality
        + 0.20 * monotonicity
    )


def _augment_trade_alpha_metrics(metrics: dict[str, Any], trades: Iterable[Any]) -> None:
    trade_list = list(trades or ())
    total_r = sum(_float(getattr(trade, "r_multiple", 0.0)) for trade in trade_list)
    total_mfe_r = sum(max(0.0, _float(getattr(trade, "mfe", 0.0)) / max(_trade_risk_per_share(trade), 1e-9)) for trade in trade_list)
    positive_mfe = sum(1 for trade in trade_list if _float(getattr(trade, "mfe", 0.0)) > 0.0)
    low_mfe = sum(1 for trade in trade_list if _float(getattr(trade, "mfe", 0.0)) <= _trade_risk_per_share(trade) * 0.10)
    rule_trade_counts: dict[str, int] = {}
    rule_total_r: dict[str, float] = {}
    for trade in trade_list:
        route = dict(getattr(trade, "route_metadata", {}) or {})
        rule_name = _score_band_rule(route)
        rule_trade_counts[rule_name] = rule_trade_counts.get(rule_name, 0) + 1
        rule_total_r[rule_name] = rule_total_r.get(rule_name, 0.0) + _float(getattr(trade, "r_multiple", 0.0))
    metrics["olr_realized_total_r"] = float(total_r)
    metrics["olr_total_mfe_r"] = float(total_mfe_r)
    metrics["olr_alpha_capture"] = float(max(0.0, total_r) / total_mfe_r) if total_mfe_r > 0.0 else 0.0
    metrics["olr_positive_mfe_share"] = positive_mfe / len(trade_list) if trade_list else 0.0
    metrics["olr_low_mfe_trade_share"] = low_mfe / len(trade_list) if trade_list else 0.0
    metrics["entry_conversion_rate"] = (
        _float(metrics.get("entry_fill_count")) / max(_float(metrics.get("selected_candidate_count")), 1.0)
        if metrics.get("selected_candidate_count")
        else 0.0
    )
    metrics["score_band_rule_trade_counts"] = dict(sorted(rule_trade_counts.items()))
    metrics["score_band_rule_realized_total_r"] = {key: float(value) for key, value in sorted(rule_total_r.items())}
    dynamic_rules = [rule for rule in rule_trade_counts if _is_dynamic_overlay_rule(rule)]
    metrics["dynamic_overlay_trade_count"] = float(sum(rule_trade_counts[rule] for rule in dynamic_rules))
    metrics["dynamic_overlay_realized_total_r"] = float(sum(rule_total_r.get(rule, 0.0) for rule in dynamic_rules))
    entry_days = {
        getattr(getattr(trade, "entry_fill_time", None), "date", lambda: None)()
        for trade in trade_list
    }
    metrics["active_trade_days"] = float(len({day for day in entry_days if day is not None}))


def _score_band_stats(labels: list[tuple[float, float, float]]) -> dict[str, float]:
    if len(labels) < 4:
        return {
            "olr_score_top_quartile_label_pct": 0.0,
            "olr_score_bottom_quartile_label_pct": 0.0,
            "olr_score_top_bottom_label_spread_pct": 0.0,
            "olr_score_monotonicity": 0.0,
            "olr_score_top_loss_share": 0.0,
        }
    ordered = sorted(labels, key=lambda item: item[0])
    bucket_size = max(1, len(ordered) // 4)
    buckets = [ordered[index * bucket_size : (index + 1) * bucket_size] for index in range(3)]
    buckets.append(ordered[3 * bucket_size :])
    bucket_returns = [mean([item[1] for item in bucket]) if bucket else 0.0 for bucket in buckets]
    top = buckets[-1]
    top_avg = bucket_returns[-1]
    bottom_avg = bucket_returns[0]
    monotonic_steps = sum(1 for left, right in zip(bucket_returns, bucket_returns[1:]) if right >= left)
    top_loss_share = _share([item[1] for item in top], lambda value: value <= 0.0)
    return {
        "olr_score_top_quartile_label_pct": float(top_avg),
        "olr_score_bottom_quartile_label_pct": float(bottom_avg),
        "olr_score_top_bottom_label_spread_pct": float(top_avg - bottom_avg),
        "olr_score_monotonicity": float(monotonic_steps / 3.0),
        "olr_score_top_loss_share": float(top_loss_share),
    }


def _format_olr_diagnostics(
    phase: int,
    metrics: dict[str, Any],
    greedy_result: GreedyResult,
    execution_context: dict[str, Any],
) -> str:
    entry_fills = _metric_float(metrics, "entry_fill_count", "total_trades")
    entry_trades = _metric_float(metrics, "entry_level_trade_count", "total_trades")
    exit_fills = _metric_float(metrics, "exit_fill_count", "exit_leg_count", "total_trades")
    entry_expected_r = _metric_float(metrics, "entry_level_expected_total_r", "expected_total_r")
    entry_profit_factor = _metric_float(metrics, "entry_level_profit_factor", "profit_factor")
    open_positions = _metric_float(metrics, "end_open_position_count", "open_position_count")
    cost_policy = dict(metrics.get("cost_policy") or execution_context.get("cost_policy") or {})
    lines = [
        "# OLR Phased Auto Diagnostics",
        "",
        f"- Generated: {_utc_now_iso()}",
        f"- Phase: {phase}",
        f"- Official training MTM return: {_pct(metrics.get('official_mtm_net_return_pct'))}",
        f"- Closed-trade net return: {_pct(metrics.get('net_return_pct'))}",
        f"- Max drawdown: {_pct(metrics.get('max_drawdown_pct', metrics.get('official_mtm_max_drawdown_pct')))}",
        f"- Final equity: {_float(metrics.get('final_equity')):.2f}",
        f"- Entry fills: {int(entry_fills)}",
        f"- Completed entry-level trades: {int(entry_trades)}",
        f"- Exit fills: {int(exit_fills)}",
        f"- End open positions: {int(open_positions)}",
        f"- Entry-level profit factor: {entry_profit_factor:.3f}",
        f"- Entry-level expected total R: {entry_expected_r:.3f}",
        f"- Alpha capture: {_float(metrics.get('olr_alpha_capture', metrics.get('mfe_capture'))):.3f}",
        f"- Discrimination quality: {_float(metrics.get('olr_discrimination_quality')):.3f}",
        f"- Negative selected label share: {_pct(metrics.get('olr_selected_negative_label_share'))}",
        f"- Score top-bottom label spread: {_pct(metrics.get('olr_score_top_bottom_label_spread_pct'))}",
        f"- Score components: {int(_float(metrics.get('phase_score_component_count')))}",
        f"- Greedy accepted features: {greedy_result.accepted_count}/{greedy_result.total_candidates}",
        f"- Shared core: {execution_context.get('shared_decision_core')}",
        f"- Fill timing: {execution_context.get('live_parity_fill_timing')}",
        f"- Cost policy: {_format_cost_policy(cost_policy)}",
        f"- Holdout excluded: {metrics.get('holdout_excluded', True)}",
        f"- Paper/live parity: {metrics.get('paper_live_parity_status', 'required_before_promotion')}",
        "",
        "## Replay Audit Evidence",
        f"- Same-bar fills: {int(_float(metrics.get('same_bar_fill_count')))}",
        f"- Forced replay closes: {int(_float(metrics.get('forced_replay_close_count')))}",
        f"- Rejected orders: {int(_float(metrics.get('rejected_order_count')))}",
        f"- Auction orders: {int(_float(metrics.get('auction_order_count')))}",
        f"- Auction nonfills: {int(_float(metrics.get('auction_nonfill_count')))}",
        f"- Open orders: {int(_float(metrics.get('open_order_count')))}",
        f"- Expired orders: {int(_float(metrics.get('expired_order_count')))}",
        f"- Decision hash: {_short_hash(metrics.get('decision_hash'))}",
        f"- Neutral action hash: {_short_hash(metrics.get('neutral_action_hash'))}",
        f"- Fill hash: {_short_hash(metrics.get('fill_hash'))}",
        f"- Trade hash: {_short_hash(metrics.get('trade_hash'))}",
        f"- Source snapshot hash: {_short_hash(metrics.get('source_snapshot_hash'))}",
        f"- Final state hash: {_short_hash(metrics.get('final_state_hash', metrics.get('state_snapshot_hash')))}",
        f"- Official metric basis: {metrics.get('official_metric_basis', execution_context.get('primary_promotion_basis'))}",
        "",
        "## Score-Band Attribution",
        f"- Dynamic overlay selected candidates: {int(_float(metrics.get('dynamic_overlay_selected_count')))}",
        f"- Dynamic overlay realized trades: {int(_float(metrics.get('dynamic_overlay_trade_count')))}",
        f"- Dynamic overlay realized total R: {_float(metrics.get('dynamic_overlay_realized_total_r')):.3f}",
        *_top_rule_lines(metrics),
        "",
        "## Execution Assumptions",
        f"- Auction mode: {execution_context.get('auction_mode')}",
        "- Official headline uses SimBroker mark-to-market final equity; closed-trade net is shown only as a reconciliation field.",
        "- Close-auction fills still require paper/live reconciliation before production promotion.",
        "",
        "## Kept Features",
        ", ".join(greedy_result.kept_features) if greedy_result.kept_features else "_None accepted._",
    ]
    return "\n".join(lines) + "\n"


def _format_cost_policy(policy: dict[str, Any]) -> str:
    if not policy:
        return "not_recorded"
    return ", ".join(f"{key}={policy[key]}" for key in sorted(policy))


def _short_hash(value: Any, length: int = 12) -> str:
    text = str(value or "")
    return text[:length] if text else "n/a"


def _top_rule_lines(metrics: dict[str, Any], *, limit: int = 6) -> list[str]:
    counts = dict(metrics.get("score_band_rule_trade_counts") or {})
    totals = dict(metrics.get("score_band_rule_realized_total_r") or {})
    if not counts:
        selected = dict(metrics.get("score_band_rule_selected_counts") or {})
        if not selected:
            return ["- Rule attribution: not available"]
        ranked_selected = sorted(selected.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[:limit]
        return [f"- Rule selected: {rule} candidates={int(count)}" for rule, count in ranked_selected]
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[:limit]
    return [
        f"- Rule realized: {rule} trades={int(count)} total_R={_float(totals.get(rule)):.3f}"
        for rule, count in ranked
    ]


def _greedy_from_state(state: PhaseState, metrics: dict[str, Any]) -> GreedyResult:
    final_phase = max(state.completed_phases, default=0)
    result = state.phase_results.get(final_phase, {}) if final_phase else {}
    return GreedyResult(
        base_score=float(result.get("base_score", 0.0)),
        final_score=float(result.get("final_score", 0.0)),
        final_mutations=dict(state.cumulative_mutations),
        kept_features=list(result.get("kept_features", [])),
        rounds=[],
        final_metrics=dict(metrics),
        total_candidates=int(result.get("total_candidates", 0)),
        accepted_count=int(result.get("accepted_count", len(result.get("kept_features", [])))),
        elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
    )


def _merged_initial_mutations(config: dict[str, Any]) -> dict[str, Any]:
    mutations = dict(BASE_MUTATIONS)
    mutations.update(_base_mutations(config, None))
    initial_path = config.get("initial_mutations_path")
    if initial_path:
        path = Path(str(initial_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        path_mutations = payload.get("mutations", payload) if isinstance(payload, dict) else {}
        if not isinstance(path_mutations, dict):
            raise ValueError(f"OLR initial_mutations_path does not contain a mutation mapping: {path}")
        mutations.update(path_mutations)
    return _canonicalize_olr_mutations(mutations)


def _canonicalize_olr_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _canonical_value(value) for key, value in sorted(dict(mutations or {}).items(), key=lambda item: str(item[0]))}


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _flatten_training_bars(dataset: OLRResearchSweepDataset) -> tuple[MarketBar, ...]:
    bars: list[MarketBar] = []
    training_dates = set(dataset.trading_dates)
    for (day, _symbol), symbol_bars in dataset.bars_by_key.items():
        if day in training_dates:
            bars.extend(symbol_bars)
    return tuple(sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol)))


def _eligible_snapshot_dates(trading_dates: Iterable[date]) -> tuple[date, ...]:
    dates = tuple(sorted(trading_dates))
    recovery_sessions = max(1, int(_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS))
    if len(dates) <= recovery_sessions:
        return ()
    return dates[:-recovery_sessions]


def _proxy_metrics_from_outcomes(
    outcomes: tuple[OLRTradeOutcome, ...],
    portfolio: dict[str, float],
) -> dict[str, Any]:
    gross_profits = [max(float(outcome.net_return_pct), 0.0) for outcome in outcomes]
    gross_losses = [abs(min(float(outcome.net_return_pct), 0.0)) for outcome in outcomes]
    total_profit = sum(gross_profits)
    total_loss = sum(gross_losses)
    total_r = sum(_proxy_r(outcome) for outcome in outcomes)
    total_mfe_r = sum(max(0.0, float(outcome.mfe_r)) for outcome in outcomes)
    deployed = _float(portfolio.get("portfolio_proxy_deployed_trade_count"), float(len(outcomes)))
    metrics: dict[str, Any] = {
        "total_trades": deployed,
        "trade_count": float(len(outcomes)),
        "official_mtm_net_return_pct": _float(portfolio.get("portfolio_proxy_net_return_pct")),
        "net_return_pct": _float(portfolio.get("portfolio_proxy_net_return_pct")),
        "official_mtm_max_drawdown_pct": abs(_float(portfolio.get("portfolio_proxy_max_drawdown_pct"))),
        "max_drawdown_pct": abs(_float(portfolio.get("portfolio_proxy_max_drawdown_pct"))),
        "gross_exposure_avg_pct": _float(portfolio.get("portfolio_proxy_avg_gross_exposure_pct")),
        "profit_factor": (total_profit / total_loss) if total_loss > 0.0 else (99.0 if total_profit > 0.0 else 0.0),
        "win_rate": _share([outcome.net_return_pct for outcome in outcomes], lambda value: value > 0.0),
        "expected_total_r": total_r,
        "avg_r": total_r / len(outcomes) if outcomes else 0.0,
        "mfe_capture": (sum(max(float(outcome.net_return_pct), 0.0) for outcome in outcomes) / sum(max(float(outcome.gross_return_pct), float(outcome.net_return_pct), 0.0) for outcome in outcomes)) if outcomes else 0.0,
        "olr_realized_total_r": total_r,
        "olr_total_mfe_r": total_mfe_r,
        "olr_alpha_capture": max(0.0, total_r) / total_mfe_r if total_mfe_r > 0.0 else 0.0,
        "same_bar_fill_count": 0.0,
        "rejected_order_count": 0.0,
        "forced_replay_close_count": 0.0,
        "end_open_position_count": 0.0,
        "open_position_count": 0.0,
        "open_order_count": 0.0,
        "entry_fill_count": deployed,
        "exit_fill_count": deployed,
        "auction_nonfill_count": _float(portfolio.get("portfolio_proxy_qty_zero_count")),
        "portfolio_proxy_net_return_pct": _float(portfolio.get("portfolio_proxy_net_return_pct")),
        "portfolio_proxy_max_drawdown_pct": _float(portfolio.get("portfolio_proxy_max_drawdown_pct")),
        "portfolio_proxy_deployed_trade_count": deployed,
        "portfolio_proxy_cash_rejected_count": _float(portfolio.get("portfolio_proxy_cash_rejected_count")),
        "portfolio_proxy_symbol_blocked_count": _float(portfolio.get("portfolio_proxy_symbol_blocked_count")),
    }
    return metrics


def _augment_proxy_trade_alpha_metrics(metrics: dict[str, Any], outcomes: tuple[OLRTradeOutcome, ...]) -> None:
    total_r = sum(_proxy_r(outcome) for outcome in outcomes)
    total_mfe_r = sum(max(0.0, float(outcome.mfe_r)) for outcome in outcomes)
    rule_trade_counts: dict[str, int] = {}
    rule_total_r: dict[str, float] = {}
    for outcome in outcomes:
        rule_name = _score_band_rule(outcome.metadata or {})
        rule_trade_counts[rule_name] = rule_trade_counts.get(rule_name, 0) + 1
        rule_total_r[rule_name] = rule_total_r.get(rule_name, 0.0) + _proxy_r(outcome)
    metrics["olr_realized_total_r"] = float(total_r)
    metrics["olr_total_mfe_r"] = float(total_mfe_r)
    metrics["olr_alpha_capture"] = float(max(0.0, total_r) / total_mfe_r) if total_mfe_r > 0.0 else 0.0
    metrics["olr_positive_mfe_share"] = _share([outcome.mfe_r for outcome in outcomes], lambda value: value > 0.0)
    metrics["olr_low_mfe_trade_share"] = _share([outcome.mfe_r for outcome in outcomes], lambda value: value <= 0.10)
    metrics["entry_conversion_rate"] = (
        _float(metrics.get("entry_fill_count")) / max(_float(metrics.get("selected_candidate_count")), 1.0)
        if metrics.get("selected_candidate_count")
        else 0.0
    )
    metrics["score_band_rule_trade_counts"] = dict(sorted(rule_trade_counts.items()))
    metrics["score_band_rule_realized_total_r"] = {key: float(value) for key, value in sorted(rule_total_r.items())}
    dynamic_rules = [rule for rule in rule_trade_counts if _is_dynamic_overlay_rule(rule)]
    metrics["dynamic_overlay_trade_count"] = float(sum(rule_trade_counts[rule] for rule in dynamic_rules))
    metrics["dynamic_overlay_realized_total_r"] = float(sum(rule_total_r.get(rule, 0.0) for rule in dynamic_rules))
    metrics["active_trade_days"] = float(len({outcome.trade_date for outcome in outcomes}))


def _score_band_rule(metadata: dict[str, Any]) -> str:
    return str(metadata.get("afternoon_score_band_rule") or "unattributed")


def _is_dynamic_overlay_rule(rule_name: str) -> bool:
    text = str(rule_name or "").lower()
    return "dynamic" in text or "sector_admission" in text


def _selection_counts(snapshots: dict[date, Any], cfg: OLRConfig) -> dict[date, int]:
    limit = max(1, int(cfg.overnight_slot_count))
    return {
        day: min(limit, len(tuple(getattr(snapshot, "candidates", ()) or ())))
        for day, snapshot in snapshots.items()
    }


def _entry_plan_from_config(cfg: OLRConfig) -> OLREntryPlan:
    payload = dict(cfg.trade_entry_plan or {})
    if payload:
        allowed = set(OLREntryPlan.__dataclass_fields__)
        return OLREntryPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLREntryPlan("", cfg.entry_mode)


def _exit_plan_from_config(cfg: OLRConfig) -> OLRExitPlan:
    payload = dict(cfg.trade_exit_plan or {})
    if payload:
        allowed = set(OLRExitPlan.__dataclass_fields__)
        return OLRExitPlan(**{key: value for key, value in payload.items() if key in allowed})
    return OLRExitPlan("", mode=cfg.exit_mode)


def _allocation_plan_from_config(cfg: OLRConfig) -> OLRAllocationPlan:
    return OLRAllocationPlan(
        name=str(cfg.allocation_mode or "allocation"),
        mode=str(cfg.allocation_mode or "selected_equal_capped"),
        target_gross_exposure=float(cfg.target_gross_exposure),
        max_position_pct=float(cfg.max_position_pct),
        min_selected=int(cfg.min_selected),
        rank_decay=float(cfg.rank_decay),
    )


def _proxy_r(outcome: OLRTradeOutcome) -> float:
    risk = max(float(outcome.risk_per_share), 1e-9)
    return (float(outcome.exit_price) - float(outcome.entry_price)) / risk


def _snapshot_hash(snapshots: dict[date, Any]) -> str:
    return stable_signature({day.isoformat(): getattr(snapshot, "artifact_hash", "") for day, snapshot in sorted(snapshots.items())})


def _next_session_map(dates: Iterable[date]) -> dict[date, date]:
    ordered = tuple(sorted(dates))
    return {day: ordered[index + 1] for index, day in enumerate(ordered[:-1])}


def _stage1_key(mutations: dict[str, Any]) -> str:
    return stable_signature(_mutation_subset(mutations, _STAGE1_PREFIXES))


def _selection_key(mutations: dict[str, Any]) -> str:
    return stable_signature(_mutation_subset(mutations, _SELECTION_PREFIXES))


def _execution_key(mutations: dict[str, Any]) -> str:
    return stable_signature(_mutation_subset(mutations, _EXECUTION_PREFIXES))


def _mutation_subset(mutations: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(mutations or {}).items():
        key_text = str(key)
        if any(key_text.startswith(prefix) for prefix in prefixes):
            out[key_text] = value
    return _canonicalize_olr_mutations(out)


def _filtered_training_bars_for_snapshots(
    dataset: OLRResearchSweepDataset,
    snapshots: dict[date, Any],
    cfg: OLRConfig,
) -> tuple[tuple[MarketBar, ...], dict[str, Any]]:
    dates = tuple(sorted(dataset.trading_dates))
    selected_pairs = _official_selected_pairs(snapshots, cfg)
    needed: set[tuple[date, str]] = set()
    for day, symbol in selected_pairs:
        needed.add((day, symbol))
        for followup_day in _official_followup_session_dates(dates, day):
            needed.add((followup_day, symbol))
    bars: list[MarketBar] = []
    for key in sorted(needed):
        bars.extend(dataset.bars_by_key.get(key, ()))
    ordered = tuple(sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol)))
    bar_scope = {
        "selected_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(selected_pairs)],
        "needed_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(needed)],
        "bar_count": len(ordered),
        "auction_exit_recovery_sessions": _OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS,
        "scope": "executable_stage2_slots_trade_date_plus_train_only_auction_exit_recovery",
    }
    return ordered, bar_scope


def _official_selected_pairs(snapshots: dict[date, Any], cfg: OLRConfig) -> set[tuple[date, str]]:
    pairs: set[tuple[date, str]] = set()
    slot_count = max(1, int(cfg.overnight_slot_count))
    min_selected = max(1, int(cfg.min_selected))
    for day, snapshot in sorted(snapshots.items()):
        candidates = tuple(getattr(snapshot, "candidates", ()) or ())
        selected = tuple(candidate for candidate in candidates[:slot_count] if bool(getattr(candidate, "tradable", True)))
        if len(selected) < min_selected:
            continue
        for candidate in selected:
            symbol = str(getattr(candidate, "symbol", "")).zfill(6)
            if symbol:
                pairs.add((day, symbol))
    return pairs


def _official_followup_session_dates(ordered_dates: Sequence[date], trade_date: date) -> tuple[date, ...]:
    ordered = tuple(sorted(ordered_dates))
    try:
        index = ordered.index(trade_date)
    except ValueError:
        return ()
    stop = min(len(ordered), index + 1 + _OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS)
    return ordered[index + 1 : stop]


def _research_snapshot_hashes(snapshots: dict[date, Any]) -> dict[str, str]:
    return {
        day.isoformat(): stable_signature(
            {
                "source_fingerprint": getattr(snapshot, "source_fingerprint", ""),
                "generated_at": getattr(snapshot, "generated_at", ""),
                "metadata": getattr(snapshot, "metadata", {}),
                "symbols": sorted(str(symbol) for symbol in getattr(snapshot, "symbols", {}).keys()),
                "sectors": sorted(str(sector) for sector in getattr(snapshot, "sectors", {}).keys()),
            }
        )
        for day, snapshot in sorted(snapshots.items())
    }


def _share(values: Iterable[float], predicate) -> float:
    seq = list(values)
    return sum(1 for value in seq if predicate(float(value))) / len(seq) if seq else 0.0


def _trade_risk_per_share(trade: Any) -> float:
    value = _float(getattr(trade, "risk_per_share", 0.0))
    if value > 0.0:
        return value
    metadata = dict(getattr(trade, "route_metadata", {}) or {})
    value = _float(metadata.get("risk_per_share"))
    if value > 0.0:
        return value
    entry = _float(getattr(trade, "entry_price", 0.0))
    stop = _float(metadata.get("initial_stop_price"))
    return max(entry - stop, entry * 0.001, 1.0)


def _cost_policy(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in ("slippage_bps", "commission_bps", "tax_bps_on_sell", "auction_adverse_bps", "auction_nonfill_rate")
        if key in config
    }


def _pct(value: Any) -> str:
    return f"{100.0 * _float(value):.3f}%"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_float(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics and metrics.get(key) not in (None, ""):
            return _float(metrics.get(key), default)
    return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
