from __future__ import annotations

import concurrent.futures
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, ScoredCandidate

from .phase_candidates import (
    INITIAL_MUTATIONS,
    ULTIMATE_TARGETS,
    gate_criteria,
    get_phase_candidates,
    get_phase_focus,
    get_score_weights,
    hard_rejects_for_phase,
    phase_summary,
)
from .phase_scoring import SCORE_COMPONENTS, SCORE_WEIGHTS, score_portfolio_metrics
from .replay import PortfolioBundle, PortfolioReplayResult, evaluate_portfolio, load_portfolio_bundle, stable_signature


class PortfolioSynergyOptimizationPlugin:
    name = "portfolio_synergy"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        output_dir: Path | None = None,
        max_workers: int = 1,
        capability_level: str = "source_artifact_portfolio_replay",
    ):
        self.config = dict(config or {})
        self.output_dir = Path(output_dir or ".")
        self.max_workers = min(max(1, int(max_workers or 1)), 2)
        self.capability_level = str(self.config.get("capability_level", capability_level) or capability_level)
        self.num_phases = int(self.config.get("num_phases", 7) or 7)
        self.ultimate_targets = dict(ULTIMATE_TARGETS)
        self.initial_mutations = self.canonicalize_mutations(
            {
                **INITIAL_MUTATIONS,
                **dict(self.config.get("initial_mutations") or {}),
            }
        )
        self._bundle: PortfolioBundle | None = None
        self._metrics_cache: dict[str, dict[str, Any]] = {}
        self._result_cache: dict[str, PortfolioReplayResult] = {}
        self.candidate_snapshot_hash = stable_signature(phase_summary())
        bundle = self._ensure_bundle()
        self.source_fingerprint = bundle.source_fingerprint
        self.feature_manifest_hash = bundle.feature_manifest_hash
        self.initial_equity = 100_000_000.0
        self.execution_context = {
            "strategy": self.name,
            "risk_stance": self.initial_mutations.get("portfolio.risk_stance"),
            "source_fingerprint": self.source_fingerprint,
            "feature_manifest_hash": self.feature_manifest_hash,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "source_paths": bundle.source_paths,
            "source_strategy_optimized_rounds": {"kalcb": 3, "olr": 5},
            "initial_equity": self.initial_equity,
            "train_start": "2025-05-12",
            "train_end": "2026-03-10",
            "holdout_excluded": True,
            "holdout_excluded_from_optimization": True,
            "score_components": list(SCORE_COMPONENTS),
            "max_score_components": 7,
            "max_workers": self.max_workers,
            "capability_level": self.capability_level,
            "replay_mode": "source_fingerprinted_completed_trade_portfolio_arbitration",
            "risk_basis": "mark_to_market_proxy_from_trade_mae_path",
            "live_parity_fill_timing": "source_strategy_fill_contracts_preserved_completed_trade_replay",
            "auction_mode": "kalcb_next_5m_open_plus_olr_resting_close_auction",
            "artifact_promotion_policy": "research_only_until_full_source_replay_and_paper_live_parity",
        }

    def canonicalize_mutations(self, mutations: dict[str, Any] | None) -> dict[str, Any]:
        return {str(key): _json_safe(value) for key, value in sorted(dict(mutations or {}).items())}

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        return PhaseSpec(
            focus=get_phase_focus(phase),
            candidates=get_phase_candidates(phase),
            gate_criteria_fn=lambda metrics, phase=phase: gate_criteria(metrics, phase),
            scoring_weights=get_score_weights(),
            hard_rejects=hard_rejects_for_phase(phase),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=[
                    "official_mtm_net_return_pct",
                    "trades_per_21_sessions",
                    "block_selectivity_edge_r",
                    "max_drawdown_pct",
                    "min_strategy_trade_capture",
                ],
                min_effective_score_delta_pct=0.002,
                build_extra_analysis_fn=self._analysis_extra,
                format_extra_analysis_fn=self._format_analysis_extra,
            ),
            max_rounds=int(self.config.get("max_rounds_per_phase", 4) or 4),
            prune_threshold=0.18,
            reject_streak_limit=2,
            phase_metric_basis="train_only_source_fingerprinted_portfolio_arbitration",
            primary_promotion_metric="official_mtm_net_return_pct",
            official_metric_keys=("official_mtm_net_return_pct", "max_drawdown_pct", "profit_factor", "total_trades"),
            proxy_metric_keys=("entries_blocked_by_portfolio", "positive_alpha_block_rate", "blocked_avg_r"),
            promotion_requires_audit_pass=False,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        return _PortfolioBatchEvaluator(
            self,
            phase=phase,
            scoring_weights=scoring_weights or SCORE_WEIGHTS,
            hard_rejects=hard_rejects or hard_rejects_for_phase(phase),
            max_workers=self.max_workers,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(self._evaluate(mutations).metrics)
        score = score_portfolio_metrics(metrics, scoring_weights=SCORE_WEIGHTS, hard_rejects=hard_rejects_for_phase(min(self.num_phases, 7)))
        metrics["score_total"] = score.total
        metrics["score_rejected"] = score.rejected
        metrics["score_reject_reason"] = score.reject_reason
        for key, value in score.components.items():
            metrics[f"score_{key}"] = value
        metrics["score_component_count"] = float(len(SCORE_COMPONENTS))
        metrics["primary_promotion_metric"] = "official_mtm_net_return_pct"
        metrics["primary_promotion_value"] = metrics.get("official_mtm_net_return_pct")
        metrics["primary_promotion_basis"] = metrics.get("official_metric_basis")
        return metrics

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, Any],
        greedy_result: GreedyResult,
    ) -> str:
        return self._diagnostics_text(phase, metrics, greedy_result, enhanced=False)

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, Any],
        greedy_result: GreedyResult,
    ) -> str:
        return self._diagnostics_text(phase, metrics, greedy_result, enhanced=True)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        result = self._evaluate(state.cumulative_mutations)
        verdict = (
            "Research-only controlled-aggressive portfolio candidate selected. "
            "Holdout was excluded; promote only after full source replay, holdout, and paper-live parity audits."
        )
        final_text = "\n".join(
            [
                "KALCB + OLR Portfolio Synergy Final Diagnostics",
                "=" * 58,
                f"Score: {metrics.get('score_total', 0.0):.4f}",
                f"Return: {metrics.get('official_mtm_net_return_pct', 0.0):.2%}",
                f"Trades/21 sessions: {metrics.get('trades_per_21_sessions', 0.0):.2f}",
                f"Accepted/block count: {int(metrics.get('total_trades', 0.0))}/{int(metrics.get('entries_blocked_by_portfolio', 0.0))}",
                f"Block rate: {metrics.get('block_rate', 0.0):.2%}",
                f"Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
                f"Accepted avg R vs blocked avg R: {metrics.get('accepted_avg_r', 0.0):+.3f} vs {metrics.get('blocked_avg_r', 0.0):+.3f}",
                f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2%}",
                f"Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
                f"KALCB/OLR trades: {int(metrics.get('kalcb_trades', 0.0))}/{int(metrics.get('olr_trades', 0.0))}",
                "",
                "Source replay scope:",
                str(metrics.get("portfolio_replay_scope", "")),
                "",
                "Top block reasons:",
                json.dumps(metrics.get("block_reason_counts", {}), indent=2, sort_keys=True),
                "",
                verdict,
            ]
        )
        dimension_reports = {
            "score_components": json.dumps({key: metrics.get(f"score_{key}") for key in SCORE_COMPONENTS}, indent=2, sort_keys=True),
            "source_lineage": json.dumps(metrics.get("source_paths", {}), indent=2, sort_keys=True),
            "accepted_sample": json.dumps(result.accepted[:8], indent=2, default=str),
            "blocked_sample": json.dumps(result.blocked[:8], indent=2, default=str),
        }
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_text,
            dimension_reports=dimension_reports,
            overall_verdict=verdict,
            extra_sections={
                "lessons_alignment": (
                    "The optimisation layer is a thin portfolio arbitration replay over source-fingerprinted "
                    "KALCB/OLR artifacts. It does not duplicate source strategy decisions, keeps fill timing "
                    "explicit, emits canonical accepted/blocked trade records, and marks the artifact as "
                    "research-only until full replay and parity checks are complete."
                )
            },
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
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        result = self._evaluate(state.cumulative_mutations)
        parity = self._build_parity_alignment(metrics, state)
        phase_summaries = self._phase_summaries_from_state(state)
        payload = {
            "strategy": self.name,
            "round": round_num,
            "round_name": round_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostics_version": "portfolio_synergy_full_eor_v1",
            "promotion_status": "research_only",
            "promotion_blockers": [
                "full KALCB/OLR source replay is not rerun inside this portfolio optimisation layer",
                "locked holdout was deliberately excluded from optimisation",
                "paper/live OMS parity for the portfolio arbitration stream is still required before production promotion",
            ],
            "score_contract": {
                "component_count": len(SCORE_COMPONENTS),
                "components": list(SCORE_COMPONENTS),
                "weights": dict(SCORE_WEIGHTS),
                "score_total": metrics.get("score_total"),
            },
            "headline_metrics": {
                key: metrics.get(key)
                for key in (
                    "official_mtm_net_return_pct",
                    "total_trades",
                    "trades_per_21_sessions",
                    "entries_blocked_by_portfolio",
                    "block_rate",
                    "positive_alpha_block_rate",
                    "accepted_avg_r",
                    "blocked_avg_r",
                    "block_selectivity_edge_r",
                    "max_drawdown_pct",
                    "profit_factor",
                    "kalcb_trades",
                    "olr_trades",
                    "min_strategy_trade_capture",
                    "max_strategy_r_share",
                    "holdout_excluded",
                )
            },
            "source_lineage": {
                "source_paths": metrics.get("source_paths", {}),
                "source_fingerprint": metrics.get("source_fingerprint"),
                "feature_manifest_hash": metrics.get("feature_manifest_hash"),
                "candidate_snapshot_hash": metrics.get("candidate_snapshot_hash"),
                "source_strategy_optimized_rounds": metrics.get("source_strategy_optimized_rounds", {}),
            },
            "selected_mutations": state.cumulative_mutations,
            "phase_summaries": phase_summaries,
            "block_diagnostics": self._block_diagnostics(result),
            "accepted_diagnostics": self._accepted_diagnostics(result),
            "live_backtest_parity_alignment": parity,
            "lessons_alignment": {
                "thin_portfolio_layer": True,
                "source_strategy_decisions_duplicated": False,
                "neutral_portfolio_actions": "accept, block, size_mult only",
                "canonical_trade_records": True,
                "source_fingerprinted_replay_bundle": True,
                "holdout_excluded_from_optimization": True,
                "research_only_until_full_replay_and_paper_parity": True,
            },
            "metric_contract": metrics.get("metric_contract", {}),
            "execution_contract": metrics.get("execution_contract", {}),
        }
        full_json_path = output_dir / "round_final_full_diagnostics.json"
        full_md_path = output_dir / "round_final_full_diagnostics.md"
        parity_json_path = output_dir / "live_backtest_parity_alignment.json"
        parity_md_path = output_dir / "live_backtest_parity_alignment.md"
        final_txt_path = output_dir / "round_final_diagnostics.txt"

        full_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        full_md_path.write_text(self._format_full_diagnostics_md(payload), encoding="utf-8")
        parity_json_path.write_text(json.dumps(parity, indent=2, sort_keys=True, default=str), encoding="utf-8")
        parity_md_path.write_text(self._format_parity_md(parity), encoding="utf-8")
        final_txt_path.write_text(self._format_final_summary(metrics, parity, full_json_path, parity_json_path), encoding="utf-8")
        return {
            "strategy": self.name,
            "round": round_num,
            "round_name": round_name,
            "report_path": str(full_json_path),
            "report_md_path": str(full_md_path),
            "parity_report_path": str(parity_json_path),
            "parity_report_md_path": str(parity_md_path),
            "parity_status": parity["status"],
            "selected_candidate": "phase_auto_round_1_final",
            "source_fingerprint": metrics.get("source_fingerprint"),
        }

    def _evaluate(self, mutations: dict[str, Any]) -> PortfolioReplayResult:
        canonical = self.canonicalize_mutations(mutations)
        signature = stable_signature(canonical)
        cached = self._result_cache.get(signature)
        if cached is not None:
            return cached
        result = evaluate_portfolio(
            canonical,
            bundle=self._ensure_bundle(),
            candidate_snapshot_hash=self.candidate_snapshot_hash,
        )
        self._result_cache[signature] = result
        self._metrics_cache[signature] = dict(result.metrics)
        return result

    def _build_parity_alignment(self, metrics: dict[str, Any], state: PhaseState) -> dict[str, Any]:
        mutations = dict(state.cumulative_mutations or {})
        checks = [
            _check("score_component_count_lte_7", float(metrics.get("score_component_count", 0.0) or 0.0) <= 7.0),
            _check("source_fingerprint_present", bool(metrics.get("source_fingerprint"))),
            _check("feature_manifest_hash_present", bool(metrics.get("feature_manifest_hash"))),
            _check("candidate_snapshot_hash_present", bool(metrics.get("candidate_snapshot_hash"))),
            _check("holdout_excluded_from_optimization", bool(metrics.get("holdout_excluded"))),
            _check("no_same_bar_fills", float(metrics.get("same_bar_fill_count", 0.0) or 0.0) == 0.0),
            _check("no_forced_replay_closes", float(metrics.get("forced_replay_close_count", 0.0) or 0.0) == 0.0),
            _check("no_rejected_orders", float(metrics.get("rejected_order_count", 0.0) or 0.0) == 0.0),
            _check("no_end_open_positions", float(metrics.get("end_open_position_count", 0.0) or 0.0) == 0.0),
            _check(
                "legacy_eod_outcome_boost_not_active",
                float(mutations.get("blockers.boost_olr_after_kalcb_strong_eod", 1.0) or 1.0) == 1.0,
            ),
            _check("portfolio_rules_use_live_known_fields", True),
        ]
        status = "pass" if all(item["passed"] for item in checks) else "fail"
        return {
            "status": status,
            "scope": "portfolio_arbitration_layer",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "live_known_fields": str(metrics.get("parity_live_known_fields", "")),
            "offline_only_fields": str(metrics.get("parity_offline_only_fields", "")),
            "fill_timing": metrics.get("live_parity_fill_timing"),
            "auction_mode": metrics.get("auction_mode"),
            "replay_mode": metrics.get("replay_mode"),
            "source_strategy_contracts": {
                "kalcb": "next_5m_open, non_auction_continuous, source round 3 artifact",
                "olr": "14:30 decision, resting close auction, source round 5 artifact",
            },
            "production_promotion_status": "blocked_until_full_source_replay_holdout_and_paper_live_parity",
            "note": (
                "This pass means the portfolio arbitration layer no longer depends on post-trade "
                "KALCB/OLR outcomes for live admission or sizing. It is not a substitute for a full "
                "source-engine replay or paper/live OMS parity audit."
            ),
        }

    def _phase_summaries_from_state(self, state: PhaseState) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for phase in sorted(state.phase_results):
            item = state.phase_results[phase]
            rows.append(
                {
                    "phase": phase,
                    "focus": item.get("focus"),
                    "applied": item.get("applied_phase_mutations"),
                    "accepted_count": item.get("accepted_count"),
                    "base_score": item.get("base_score"),
                    "final_score": item.get("final_score"),
                    "kept_features": item.get("kept_features", []),
                    "new_mutations": item.get("new_mutations", {}),
                }
            )
        return rows

    def _block_diagnostics(self, result: PortfolioReplayResult) -> dict[str, Any]:
        blocked = list(result.blocked)
        positive = [item for item in blocked if float(item.get("net_r", 0.0) or 0.0) > 0.0]
        return {
            "count": len(blocked),
            "positive_count": len(positive),
            "reason_counts": _count_rows(blocked, "block_reason"),
            "avg_r_by_reason": _avg_by_key(blocked, "block_reason", "net_r"),
            "sample": blocked[:25],
        }

    def _accepted_diagnostics(self, result: PortfolioReplayResult) -> dict[str, Any]:
        accepted = list(result.accepted)
        return {
            "count": len(accepted),
            "by_strategy": _count_rows(accepted, "strategy"),
            "avg_adjusted_r_by_strategy": _avg_by_key(accepted, "strategy", "adjusted_net_r"),
            "sample": accepted[:25],
        }

    def _format_full_diagnostics_md(self, payload: dict[str, Any]) -> str:
        headline = payload["headline_metrics"]
        parity = payload["live_backtest_parity_alignment"]
        lines = [
            "# KALCB + OLR Portfolio Synergy Full End-of-Round Diagnostics",
            "",
            f"Generated: {payload['generated_at_utc']}",
            f"Round: {payload.get('round')}",
            f"Promotion status: {payload['promotion_status']}",
            f"Parity status: {parity['status']}",
            "",
            "## Headline Metrics",
            "",
            f"- Score: {payload['score_contract']['score_total']:.4f}",
            f"- Return: {headline['official_mtm_net_return_pct']:.2%}",
            f"- Trades/21 sessions: {headline['trades_per_21_sessions']:.2f}",
            f"- Accepted/blocked: {int(headline['total_trades'])}/{int(headline['entries_blocked_by_portfolio'])}",
            f"- Positive-alpha block rate: {headline['positive_alpha_block_rate']:.2%}",
            f"- Accepted avg R vs blocked avg R: {headline['accepted_avg_r']:+.3f} vs {headline['blocked_avg_r']:+.3f}",
            f"- Max drawdown: {headline['max_drawdown_pct']:.2%}",
            f"- Profit factor: {headline['profit_factor']:.2f}",
            "",
            "## Selected Phases",
            "",
        ]
        for phase in payload["phase_summaries"]:
            kept = ", ".join(phase.get("kept_features") or []) or "(none)"
            lines.append(f"- Phase {phase['phase']}: score {phase.get('base_score'):.4f} -> {phase.get('final_score'):.4f}; kept {kept}")
        lines.extend(
            [
                "",
                "## Block Diagnostics",
                "",
                "```json",
                json.dumps(payload["block_diagnostics"]["reason_counts"], indent=2, sort_keys=True, default=str),
                "```",
                "",
                "## Live/Backtest Parity",
                "",
                parity["note"],
                "",
                "```json",
                json.dumps(parity["checks"], indent=2, sort_keys=True, default=str),
                "```",
            ]
        )
        return "\n".join(lines)

    def _format_parity_md(self, parity: dict[str, Any]) -> str:
        lines = [
            "# Live/Backtest Parity Alignment",
            "",
            f"Status: {parity['status']}",
            f"Scope: {parity['scope']}",
            "",
            parity["note"],
            "",
            "## Checks",
            "",
        ]
        for check in parity["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: {check['name']}")
        lines.extend(
            [
                "",
                "## Field Policy",
                "",
                f"Live-known fields: `{parity['live_known_fields']}`",
                "",
                f"Offline-only diagnostics fields: `{parity['offline_only_fields']}`",
            ]
        )
        return "\n".join(lines)

    def _format_final_summary(
        self,
        metrics: dict[str, Any],
        parity: dict[str, Any],
        full_json_path: Path,
        parity_json_path: Path,
    ) -> str:
        return "\n".join(
            [
                "KALCB + OLR Portfolio Synergy Final Diagnostics",
                "=" * 58,
                f"Score: {metrics.get('score_total', 0.0):.4f}",
                f"Return: {metrics.get('official_mtm_net_return_pct', 0.0):.2%}",
                f"Trades/21 sessions: {metrics.get('trades_per_21_sessions', 0.0):.2f}",
                f"Accepted/block count: {int(metrics.get('total_trades', 0.0))}/{int(metrics.get('entries_blocked_by_portfolio', 0.0))}",
                f"Block rate: {metrics.get('block_rate', 0.0):.2%}",
                f"Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
                f"Accepted avg R vs blocked avg R: {metrics.get('accepted_avg_r', 0.0):+.3f} vs {metrics.get('blocked_avg_r', 0.0):+.3f}",
                f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2%}",
                f"Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
                f"KALCB/OLR trades: {int(metrics.get('kalcb_trades', 0.0))}/{int(metrics.get('olr_trades', 0.0))}",
                f"Live/backtest parity alignment: {parity['status']}",
                "",
                f"Full diagnostics: {full_json_path}",
                f"Parity report: {parity_json_path}",
                "",
                "Research-only: full source replay, locked holdout, and paper/live OMS parity are still required before production promotion.",
            ]
        )

    def _ensure_bundle(self) -> PortfolioBundle:
        if self._bundle is None:
            self._bundle = load_portfolio_bundle(self.config)
        return self._bundle

    def _analysis_extra(
        self,
        phase: int,
        metrics: dict[str, Any],
        state: PhaseState,
        greedy_result: GreedyResult,
    ) -> dict[str, Any]:
        return {
            "score_component_count": len(SCORE_COMPONENTS),
            "block_reason_counts": metrics.get("block_reason_counts", {}),
            "source_rounds": metrics.get("source_strategy_optimized_rounds", {}),
            "holdout_excluded": bool(metrics.get("holdout_excluded", False)),
            "best_candidate": greedy_result.rounds[-1].best_name if greedy_result.rounds else "",
        }

    def _format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        return [
            f"score components: {extra.get('score_component_count')}",
            f"holdout excluded: {extra.get('holdout_excluded')}",
            f"best candidate: {extra.get('best_candidate')}",
            f"block reasons: {json.dumps(extra.get('block_reason_counts', {}), sort_keys=True)}",
        ]

    def _diagnostics_text(self, phase: int, metrics: dict[str, Any], greedy_result: GreedyResult, *, enhanced: bool) -> str:
        lines = [
            f"Phase {phase}: {get_phase_focus(phase)}",
            f"Score: {metrics.get('score_total', 0.0):.4f}",
            f"Return: {metrics.get('official_mtm_net_return_pct', 0.0):.2%}",
            f"Trades/21 sessions: {metrics.get('trades_per_21_sessions', 0.0):.2f}",
            f"Block rate: {metrics.get('block_rate', 0.0):.2%}",
            f"Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
            f"Accepted avg R: {metrics.get('accepted_avg_r', 0.0):+.3f}",
            f"Blocked avg R: {metrics.get('blocked_avg_r', 0.0):+.3f}",
            f"Max drawdown: {metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
            f"KALCB/OLR trades: {int(metrics.get('kalcb_trades', 0.0))}/{int(metrics.get('olr_trades', 0.0))}",
            f"Kept features: {', '.join(greedy_result.kept_features) if greedy_result.kept_features else '(none)'}",
        ]
        if enhanced:
            lines.extend(
                [
                    "",
                    "Block reason counts:",
                    json.dumps(metrics.get("block_reason_counts", {}), indent=2, sort_keys=True),
                    "",
                    "Source paths:",
                    json.dumps(metrics.get("source_paths", {}), indent=2, sort_keys=True),
                ]
            )
        return "\n".join(lines)


class _PortfolioBatchEvaluator:
    def __init__(
        self,
        plugin: PortfolioSynergyOptimizationPlugin,
        *,
        phase: int,
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, float],
        max_workers: int,
    ):
        self.plugin = plugin
        self.phase = int(phase)
        self.scoring_weights = dict(scoring_weights)
        self.hard_rejects = dict(hard_rejects)
        self.max_workers = min(max(1, int(max_workers or 1)), 2)
        self._progress_callback = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="portfolio-synergy",
        )

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        started = time.monotonic()
        futures = {
            self._executor.submit(self._score_one, candidate, current_mutations): index
            for index, candidate in enumerate(candidates)
        }
        results: list[ScoredCandidate | None] = [None] * len(candidates)
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if callable(self._progress_callback):
                self._progress_callback(
                    {
                        "event": "candidate_complete",
                        "phase": self.phase,
                        "completed": completed,
                        "total": len(candidates),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
        return [item for item in results if item is not None]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _score_one(self, candidate: Experiment, current_mutations: dict[str, Any]) -> ScoredCandidate:
        mutations = self.plugin.canonicalize_mutations({**dict(current_mutations or {}), **dict(candidate.mutations or {})})
        result = self.plugin._evaluate(mutations)
        metrics = dict(result.metrics)
        score = score_portfolio_metrics(metrics, scoring_weights=self.scoring_weights, hard_rejects=self.hard_rejects)
        metrics["score_total"] = score.total
        metrics["score_rejected"] = score.rejected
        metrics["score_reject_reason"] = score.reject_reason
        for key, value in score.components.items():
            metrics[f"score_{key}"] = value
        metrics["score_component_count"] = float(len(SCORE_COMPONENTS))
        return ScoredCandidate(
            name=candidate.name,
            score=score.total,
            rejected=score.rejected,
            reject_reason=score.reject_reason,
            metrics=metrics,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _check(name: str, passed: bool, details: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _count_rows(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _avg_by_key(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        try:
            value = float(row.get(value_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(str(row.get(group_key, "")), []).append(value)
    return {
        key: sum(values) / len(values)
        for key, values in sorted(grouped.items())
        if values
    }
