from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.auto.shared.cache_keys import build_cache_key
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GreedyResult, PhaseAnalysis, ScoredCandidate
from backtests.strategies.common.plugin_base import SharedStrategyPluginMixin, attach_official_metric_contract, build_execution_contract
from strategy_kalcb.config import KALCB_CORE_VERSION

from .diagnostics import format_kalcb_diagnostics
from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import IMMUTABLE_SCORE_COMPONENTS, PHASE_HARD_REJECTS, ULTIMATE_TARGETS, gate_criteria, kalcb_reject_reason, score_kalcb_phase
from .runner import run_kalcb_backtest

_KALCB_HEARTBEAT_SECONDS = 30.0
_KALCB_PER_CANDIDATE_TIMEOUT_SECONDS = 180.0
_KALCB_MINIMUM_TIMEOUT_SECONDS = 240.0
_KALCB_MAX_EVAL_BATCH_SIZE = 8


class KALCBOptimizationPlugin(SharedStrategyPluginMixin):
    name = "kalcb"
    num_phases = 6
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = dict(BASE_MUTATIONS)
    default_scoring_weights = dict(IMMUTABLE_SCORE_COMPONENTS)
    requires_full_diagnostics = True

    def __init__(self, config: dict[str, Any] | None = None, *, output_dir: Path | None = None, max_workers: int | None = 1, capability_level: str = "real_replay"):
        self.config = dict(config or {})
        self.config.setdefault("capability_level", capability_level)
        self.config.setdefault("live_parity_fill_timing", "next_5m_open")
        self.config.setdefault("auction_mode", "non_auction_continuous")
        self.config.setdefault("artifact_promotion_policy", "research_only_until_oos_and_paper_parity")
        self.config.setdefault("promotion_requires_audit_pass", True)
        self.max_workers = max_workers
        self.capability_level = self.config.get("capability_level", capability_level)
        self.output_dir = Path(output_dir) if output_dir else None
        self.initial_mutations = _merged_initial_mutations(self.config)
        baseline = run_kalcb_backtest(self.config, self.initial_mutations)
        self.source_fingerprint = baseline.source_fingerprint
        self.feature_manifest_hash = baseline.feature_bundle_hash
        cache_execution_identity = {
            "feature_manifest_hash": self.feature_manifest_hash,
            "strategy_core_version": KALCB_CORE_VERSION,
            "candidate_snapshot_hash": baseline.candidate_snapshot_hash,
            "capability_level": self.capability_level,
            "replay_mode": baseline.metrics.get("replay_mode", self.config.get("capability_level", "")),
            "initial_equity": self.config.get("initial_equity", ""),
            "cost_policy": _cost_policy(self.config),
            "fill_timing": self.config["live_parity_fill_timing"],
            "auction_mode": self.config["auction_mode"],
            "official_proxy_mode": "direct_official_replay",
            "date_window": {
                "start": self.config.get("train_start", self.config.get("start_date", "")),
                "end": self.config.get("train_end", self.config.get("end_date", "")),
                "holdout_start": self.config.get("holdout_start", ""),
                "holdout_end": self.config.get("holdout_end", ""),
            },
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        }
        raw_metric_cache_key = build_cache_key(
            "kalcb.raw_metrics",
            source_fingerprint=self.source_fingerprint,
            mutations=self.initial_mutations,
            extra=cache_execution_identity,
        )
        phase_score_cache_key = build_cache_key(
            "kalcb.phase_score",
            source_fingerprint=self.source_fingerprint,
            mutations=self.initial_mutations,
            extra={
                **cache_execution_identity,
                "default_scoring_weights": self.default_scoring_weights,
            },
        )
        self.execution_context = {
            "shared_decision_core": "live_shared_core",
            "strategy_core_version": KALCB_CORE_VERSION,
            "source_fingerprint": baseline.source_fingerprint,
            "feature_manifest_hash": baseline.feature_bundle_hash,
            "candidate_snapshot_hash": baseline.candidate_snapshot_hash,
            "live_parity_fill_timing": self.config["live_parity_fill_timing"],
            "auction_mode": self.config["auction_mode"],
            "initial_equity": self.config.get("initial_equity", ""),
            "cost_policy": _cost_policy(self.config),
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "artifact_promotion_policy": self.config["artifact_promotion_policy"],
            "account_scope": self.config.get("account_scope", "krx_cash_long_only"),
            "diagnostics_version": "kalcb-diagnostics-v2",
            "phase_analyzer_version": "shared-v1",
            "raw_metric_cache_key": raw_metric_cache_key,
            "phase_score_cache_key": phase_score_cache_key,
            "resume_checkpoint_id": "",
        }
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._remember_backtest_result(self.initial_mutations, baseline)

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
            max_rounds=3,
            prune_threshold=0.10,
            reject_streak_limit=2,
            phase_metric_basis="direct_official_replay",
            primary_promotion_metric="official_mtm_net_return_pct",
            official_metric_keys=("official_mtm_net_return_pct",),
            promotion_requires_audit_pass=True,
        )

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights: dict[str, float] | None = None, hard_rejects: dict[str, float] | None = None):
        from .worker import init_worker, score_candidate

        base_result = self._remember_backtest_result(cumulative_mutations, run_kalcb_backtest(self.config, cumulative_mutations))
        reject = kalcb_reject_reason(phase, base_result.metrics, hard_rejects)
        baseline = ScoredCandidate("__baseline__", 0.0 if reject else score_kalcb_phase(phase, base_result.metrics, scoring_weights), bool(reject), reject, base_result.metrics)
        return self._wrap_cached_evaluator(
            phase=phase,
            cumulative_mutations=cumulative_mutations,
            scoring_weights=scoring_weights,
            hard_rejects=hard_rejects,
            init_worker=init_worker,
            score_candidate=score_candidate,
            initargs=(self.config, phase, hard_rejects, scoring_weights),
            heartbeat_seconds=_KALCB_HEARTBEAT_SECONDS,
            per_candidate_timeout_seconds=_KALCB_PER_CANDIDATE_TIMEOUT_SECONDS,
            minimum_timeout_seconds=_KALCB_MINIMUM_TIMEOUT_SECONDS,
            max_eval_batch_size=_KALCB_MAX_EVAL_BATCH_SIZE,
            description=f"kalcb phase {phase}",
            baseline_result=baseline,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        result = self._last_result_for_mutations(mutations)
        if result is None:
            result = self._remember_backtest_result(mutations, run_kalcb_backtest(self.config, mutations))
        return attach_official_metric_contract(
            dict(result.metrics),
            requires_audit_pass=True,
            audit_pass=bool(
                result.metrics.get("same_bar_fill_count", 0.0) == 0.0
                and result.metrics.get("forced_replay_close_count", 0.0) == 0.0
                and result.metrics.get("rejected_order_count", 0.0) == 0.0
            ),
            audit_status="direct_official_replay",
            official_replay_pass=True,
            execution_contract=build_execution_contract(self, result.metrics),
        )

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        del phase, state, metrics, greedy_result
        return format_kalcb_diagnostics(self._last_result)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result) + "\nEnhanced checks: completed-bar causality, KIS WS slice, REST no-op fallback, and ALCB rejection funnel parity."

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final = self._remember_backtest_result(state.cumulative_mutations, run_kalcb_backtest(self.config, state.cumulative_mutations))
        return EndOfRoundArtifacts(
            final_diagnostics_text=format_kalcb_diagnostics(final),
            dimension_reports={
                "signal_extraction": "KALCB daily candidates use prior completed OHLCV and enforce ws_budget before bar replay.",
                "entry_mechanism": "OR/PDH/combined breakouts submit neutral entry actions after completed 5m bars for next_5m_open fills.",
                "trade_management": "Quick exit, failure-stop tightening, MFE conviction, flow reversal, and adaptive trail route through SimBroker/OMS actions.",
                "live_constraints": "KIS websocket capacity and EGW00201 REST cooldown are explicit live adapter constraints.",
                "promotion": "Research-only until Korean OOS, artifact hygiene, and paper parity evidence are attached.",
            },
            overall_verdict="KALCB replay completed through the shared live core; promotion requires source-fingerprinted Korean holdout and paper/live decision stream parity.",
        )

    def write_full_diagnostics(
        self,
        state: PhaseState,
        output_dir: Path,
        *,
        round_num: int | None = None,
        round_name: str = "",
    ) -> dict[str, Any]:
        from .full_diagnostics import write_kalcb_optimization_full_diagnostics

        return write_kalcb_optimization_full_diagnostics(
            config=self.config,
            state=state,
            output_dir=Path(output_dir),
            round_num=round_num,
            round_name=round_name,
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        del phase
        gaps = []
        if self.capability_level == "synthetic":
            gaps.append("Synthetic KALCB runs are fixture-only and must not be used for baseline promotion.")
        if metrics.get("same_bar_fill_count", 0.0) > 0:
            gaps.append("Same-bar fills observed; live-parity promotion is blocked.")
        if metrics.get("active_symbol_max", 0.0) > float(self.initial_mutations.get("kalcb.session.ws_budget", 10)):
            gaps.append("Active symbol count exceeds configured websocket slice.")
        return gaps

    def suggest_experiments(self, phase: int, metrics: dict[str, float], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del weaknesses, state
        if phase == 2 and metrics.get("entry_rejection_count", 0.0) <= 0:
            return [Experiment("rvol_cap_tighter_for_funnel", {"kalcb.entry.rvol_max": 4.0})]
        if phase == 5 and metrics.get("total_trades", 0.0) < 20:
            return [Experiment("ws_budget_14_frequency_probe", {"kalcb.session.ws_budget": 14})]
        return []

    def redesign_scoring_weights(self, phase: int, current_weights: dict[str, float] | None, analysis: PhaseAnalysis, gate_result) -> dict[str, float] | None:
        del phase, current_weights, analysis, gate_result
        return None


def _merged_initial_mutations(config: dict[str, Any]) -> dict[str, Any]:
    mutations = dict(BASE_MUTATIONS)
    config_initial = config.get("initial_mutations")
    if isinstance(config_initial, dict):
        mutations.update(config_initial)
    return mutations


def _cost_policy(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in ("slippage_bps", "commission_bps", "tax_bps_on_sell")
        if key in config
    }
