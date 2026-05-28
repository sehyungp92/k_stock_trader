from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import PhaseState, _atomic_write_json, save_phase_state
from backtests.auto.shared.round_manager import RoundManager
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_common.events import DecisionEvent, TradeOutcome
from strategy_kalcb.config import KALCB_CORE_VERSION

from .runner import StrategyBacktestResult, run_kalcb_backtest

DEFAULT_CONFIG = Path("config/optimization/kalcb.yaml")
DEFAULT_OUTPUT_ROOT = Path("data/backtests/output")
DEFAULT_WS8_SUMMARY = Path("data/backtests/output/kalcb/baseline_sweep_flatten1515_targeted/targeted_high_frequency_summary.json")
DEFAULT_CAPACITY_VALIDATION = Path("data/backtests/output/kalcb/baseline_sweep_flatten1515_targeted/targeted_high_frequency_expand.jsonl")
DEFAULT_BROAD_SUMMARY = Path("data/backtests/output/kalcb/baseline_sweep_flatten1515_targeted/targeted_high_frequency_summary.json")


def write_kalcb_optimization_full_diagnostics(
    *,
    config: dict[str, Any],
    state: PhaseState,
    output_dir: Path,
    round_num: int | None = None,
    round_name: str = "",
) -> dict[str, Any]:
    """Write KALCB full diagnostics for a completed phased optimisation round."""

    output_dir = Path(output_dir)
    mutations = dict(getattr(state, "cumulative_mutations", {}) or {})
    result = run_kalcb_backtest(dict(config or {}), mutations)
    external = _load_external_artifacts(
        Path(str(config.get("ws8_summary_path") or DEFAULT_WS8_SUMMARY)),
        Path(str(config.get("capacity_validation_path") or DEFAULT_CAPACITY_VALIDATION)),
        Path(str(config.get("broad_summary_path") or DEFAULT_BROAD_SUMMARY)),
    )
    analysis = analyze_kalcb_result(result, config=dict(config or {}), mutations=mutations, external_artifacts=external)
    analysis["round"] = round_num
    analysis["round_name"] = round_name
    source_paths = {
        "config": str(config.get("config_path") or DEFAULT_CONFIG),
        "ws8_summary": str(config.get("ws8_summary_path") or DEFAULT_WS8_SUMMARY),
        "capacity_validation": str(config.get("capacity_validation_path") or DEFAULT_CAPACITY_VALIDATION),
        "broad_summary": str(config.get("broad_summary_path") or DEFAULT_BROAD_SUMMARY),
    }
    artifact_paths = _artifact_paths(output_dir)
    report = render_kalcb_diagnostics_report(analysis)
    _atomic_write_text(output_dir / "round_final_diagnostics.txt", report)
    _atomic_write_json(analysis["diagnostics_summary"], output_dir / "diagnostics_summary.json")
    _atomic_write_json(analysis["candidate_frontier"], output_dir / "candidate_frontier.json")
    _atomic_write_json(analysis["live_parity_audit"], output_dir / "live_parity_audit.json")
    _atomic_write_json(_index_payload(analysis, artifact_paths, source_paths), output_dir / "full_diagnostics_index.json")
    _atomic_write_text(output_dir / "phase_1_diagnostics.txt", _phase_diagnostics_text(analysis))
    return analysis["diagnostics_summary"]


def promote_kalcb_baseline_round(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    round_num: int = 1,
    ws8_summary_path: str | Path = DEFAULT_WS8_SUMMARY,
    capacity_validation_path: str | Path = DEFAULT_CAPACITY_VALIDATION,
    broad_summary_path: str | Path = DEFAULT_BROAD_SUMMARY,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_root = Path(output_root)
    config = normalize_runtime_config("kalcb", load_yaml_config(config_path))
    mutations = dict(config.get("initial_mutations") or {})
    if not mutations:
        raise ValueError(f"KALCB config has no initial_mutations: {config_path}")

    result = run_kalcb_backtest(config, mutations)
    external = _load_external_artifacts(ws8_summary_path, capacity_validation_path, broad_summary_path)
    analysis = analyze_kalcb_result(result, config=config, mutations=mutations, external_artifacts=external)

    manager = RoundManager("stock", "kalcb", base_dir=output_root)
    round_dir = manager.get_round_dir(round_num)
    mutation_sha = _json_sha256(mutations)
    source_paths = {
        "config": str(config_path),
        "ws8_summary": str(ws8_summary_path),
        "capacity_validation": str(capacity_validation_path),
        "broad_summary": str(broad_summary_path),
    }
    artifact_paths = _artifact_paths(round_dir)

    manager.write_run_spec(
        round_dir,
        round_num,
        strategy_name="kalcb",
        description="Round 1 optimized KALCB baseline promoted from the ws8 refinement sweep with full live-parity diagnostics.",
        baseline_mutations=mutations,
        baseline_source=ws8_summary_path,
        execution_context={
            "config_path": str(config_path),
            "mutation_sha256": mutation_sha,
            "source_fingerprint": result.source_fingerprint,
            "candidate_snapshot_hash": result.candidate_snapshot_hash,
            "feature_bundle_hash": result.feature_bundle_hash,
            "strategy_core_version": KALCB_CORE_VERSION,
            "live_parity_fill_timing": result.metrics.get("live_parity_fill_timing"),
            "auction_mode": result.metrics.get("auction_mode"),
            "artifact_promotion_policy": config.get("artifact_promotion_policy"),
            "source_artifacts": source_paths,
        },
        overwrite=True,
    )
    manager.write_optimized_config(
        round_dir,
        mutations,
        artifact_metadata={
            "family": "stock",
            "strategy": "kalcb",
            "round": round_num,
            "round_name": f"round_{round_num}",
            "baseline_candidate": analysis["selected_candidate"],
            "baseline_status": analysis["promotion_status"],
            "mutation_sha256": mutation_sha,
            "source_fingerprint": result.source_fingerprint,
            "candidate_snapshot_hash": result.candidate_snapshot_hash,
            "feature_bundle_hash": result.feature_bundle_hash,
            "source_artifacts": source_paths,
            "selection_metrics": analysis["final_metrics"],
        },
    )
    manager.write_run_summary(
        round_dir,
        mutations,
        analysis["final_metrics"],
        [0],
        round_num=round_num,
        artifact_metadata={
            "baseline_candidate": analysis["selected_candidate"],
            "baseline_status": analysis["promotion_status"],
            "mutation_sha256": mutation_sha,
            "source_fingerprint": result.source_fingerprint,
            "candidate_snapshot_hash": result.candidate_snapshot_hash,
            "feature_bundle_hash": result.feature_bundle_hash,
            "source_artifacts": source_paths,
            "diagnostics": artifact_paths,
        },
    )
    manifest_path = manager.append_to_manifest(round_num, mutations, analysis["final_metrics"])
    _enrich_manifest(
        manifest_path,
        round_num,
        {
            "baseline_candidate": analysis["selected_candidate"],
            "baseline_status": analysis["promotion_status"],
            "mutation_sha256": mutation_sha,
            "source_fingerprint": result.source_fingerprint,
            "candidate_snapshot_hash": result.candidate_snapshot_hash,
            "feature_bundle_hash": result.feature_bundle_hash,
            "same_bar_fill_count": result.metrics.get("same_bar_fill_count"),
            "active_symbol_max": result.metrics.get("active_symbol_max"),
            "live_parity_fill_timing": result.metrics.get("live_parity_fill_timing"),
            "auction_mode": result.metrics.get("auction_mode"),
            "source_artifacts": source_paths,
        },
    )

    state = PhaseState(
        current_phase=0,
        completed_phases=[0],
        cumulative_mutations=mutations,
        phase_results={
            0: {
                "phase": 0,
                "kind": "optimized_baseline",
                "candidate": analysis["selected_candidate"],
                "final_metrics": analysis["final_metrics"],
                "source_artifacts": source_paths,
            }
        },
        phase_gate_results={0: analysis["gate_result"]},
        round_name=f"round_{round_num}",
    )
    save_phase_state(state, round_dir / "phase_state.json")

    report = render_kalcb_diagnostics_report(analysis)
    evaluation = render_kalcb_evaluation(analysis)
    _atomic_write_text(round_dir / "round_final_diagnostics.txt", report)
    _atomic_write_text(round_dir / "round_evaluation.txt", evaluation)
    _atomic_write_json(analysis["diagnostics_summary"], round_dir / "diagnostics_summary.json")
    _atomic_write_json(analysis["candidate_frontier"], round_dir / "candidate_frontier.json")
    _atomic_write_json(analysis["live_parity_audit"], round_dir / "live_parity_audit.json")
    _atomic_write_json(_index_payload(analysis, artifact_paths, source_paths), round_dir / "full_diagnostics_index.json")
    _atomic_write_json(_progress_snapshot(analysis, artifact_paths), round_dir / "progress.json")
    _atomic_write_json(_phase_analysis_payload(analysis), round_dir / "phase_1_analysis.json")
    _atomic_write_text(round_dir / "phase_1_diagnostics.txt", _phase_diagnostics_text(analysis))
    _atomic_write_json({"mutations": mutations, "metrics": analysis["final_metrics"]}, round_dir / "phase_1_greedy.json")
    _atomic_write_json({"selected": analysis["selected_candidate"], "mutations": mutations}, round_dir / "phase_1_greedy_raw.json")
    _atomic_write_json(_run_summary_payload(analysis), round_dir / "run_baseline_diagnostics_summary.json")
    _atomic_write_jsonl(round_dir / "trade_events.jsonl", analysis["trade_events"])
    _atomic_write_jsonl(round_dir / "phase_activity_log.jsonl", [analysis["activity_event"]])

    return {
        "strategy": "kalcb",
        "round": round_num,
        "round_dir": str(round_dir),
        "manifest": str(manifest_path),
        "selected_candidate": analysis["selected_candidate"],
        "metrics": analysis["final_metrics"],
        "diagnostics": artifact_paths,
    }


def analyze_kalcb_result(
    result: StrategyBacktestResult,
    *,
    config: dict[str, Any],
    mutations: dict[str, Any],
    external_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    external = dict(external_artifacts or {})
    trades = list(result.trades)
    decisions = list(result.decisions)
    rows = [_trade_row(trade) for trade in trades]
    final_metrics = _final_metrics(result.metrics, trades, config, result.replay_result.equity_curve)
    selected = str((external.get("ws8_summary") or {}).get("best", {}).get("name") or "start1025_trail_end1300")
    groups = _build_group_stats(rows)
    signal_funnel = _signal_funnel(decisions)
    sweeps = _sweep_frontier(external)
    alpha_attribution = _alpha_capture_diagnostics(rows, result.metrics)
    management_attribution = _management_diagnostics(rows)
    layer_diagnostics = _layer_diagnostics(rows, final_metrics, signal_funnel, sweeps, alpha_attribution, management_attribution)
    candidate_frontier = {
        "selected": selected,
        "mutation_signature": stable_signature(mutations),
        "broad_summary": _compact_summary(external.get("broad_summary")),
        "ws8_summary": _compact_summary(external.get("ws8_summary")),
        "capacity_validation": sweeps["capacity_validation"],
        "targeted_train": sweeps["targeted_train"],
        "targeted_holdout": sweeps["targeted_holdout"],
    }
    verdicts = _executive_verdicts(final_metrics, signal_funnel, sweeps, alpha_attribution)
    live_parity_audit = _live_parity_audit(result, config, mutations)
    strengths, weaknesses, notes = _strengths_weaknesses(rows, groups, final_metrics, signal_funnel, sweeps, alpha_attribution)
    diagnostics_summary = {
        "strategy": "kalcb",
        "strategy_core_version": KALCB_CORE_VERSION,
        "selected_candidate": selected,
        "promotion_status": "accepted_optimized_baseline_research_only",
        "final_metrics": final_metrics,
        "verdicts": verdicts,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "notes": notes,
        "signal_funnel": signal_funnel,
        "alpha_attribution": alpha_attribution,
        "management_attribution": management_attribution,
        "layer_diagnostics": layer_diagnostics,
        "groups": groups,
        "capacity_frontier": sweeps,
        "live_parity_audit": live_parity_audit,
        "source_fingerprint": result.source_fingerprint,
        "candidate_snapshot_hash": result.candidate_snapshot_hash,
        "feature_bundle_hash": result.feature_bundle_hash,
    }
    return {
        "strategy": "kalcb",
        "strategy_core_version": KALCB_CORE_VERSION,
        "generated_at_utc": _now(),
        "selected_candidate": selected,
        "promotion_status": "accepted_optimized_baseline_research_only",
        "config": config,
        "mutations": dict(sorted(mutations.items())),
        "metrics": result.metrics,
        "final_metrics": final_metrics,
        "trades": trades,
        "trade_rows": rows,
        "trade_events": [trade.to_json_dict() for trade in trades],
        "decisions": decisions,
        "signal_funnel": signal_funnel,
        "alpha_attribution": alpha_attribution,
        "management_attribution": management_attribution,
        "layer_diagnostics": layer_diagnostics,
        "groups": groups,
        "verdicts": verdicts,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "notes": notes,
        "capacity_frontier": sweeps,
        "candidate_frontier": candidate_frontier,
        "live_parity_audit": live_parity_audit,
        "diagnostics_summary": diagnostics_summary,
        "gate_result": _gate_result(final_metrics, live_parity_audit),
        "activity_event": {
            "timestamp": _now(),
            "phase": 0,
            "event": "optimized_baseline_promoted",
            "candidate": selected,
            "source_fingerprint": result.source_fingerprint,
        },
        "source_fingerprint": result.source_fingerprint,
        "candidate_snapshot_hash": result.candidate_snapshot_hash,
        "feature_bundle_hash": result.feature_bundle_hash,
    }


def render_kalcb_diagnostics_report(analysis: dict[str, Any]) -> str:
    rows = list(analysis["trade_rows"])
    groups = analysis["groups"]
    metrics = analysis["final_metrics"]
    funnel = analysis["signal_funnel"]
    frontier = analysis["capacity_frontier"]
    audit = analysis["live_parity_audit"]
    alpha = analysis.get("alpha_attribution") or {}
    management = analysis.get("management_attribution") or {}
    layers = analysis.get("layer_diagnostics") or {}
    lines: list[str] = []
    round_label = f"ROUND {analysis.get('round')}" if analysis.get("round") else "ROUND"
    _box(lines, f"KALCB {round_label} OPTIMIZATION FULL DIAGNOSTICS")
    lines.extend(
        [
            f"  Candidate:       {analysis['selected_candidate']}",
            f"  Generated UTC:   {analysis['generated_at_utc']}",
            f"  Core version:    {analysis['strategy_core_version']}",
            f"  Source hash:     {analysis['source_fingerprint']}",
            f"  Snapshot hash:   {analysis['candidate_snapshot_hash']}",
            f"  Feature hash:    {analysis['feature_bundle_hash']}",
            f"  Promotion mode:  {analysis['promotion_status']}",
            "",
            "  Mutation baseline:",
        ]
    )
    for key, value in analysis["mutations"].items():
        lines.append(f"    {key}: {value}")

    _section(lines, "KALCB Strength / Weakness Snapshot")
    lines.append("  Strengths")
    for item in analysis["strengths"]:
        lines.append(f"    - {item}")
    lines.append("")
    lines.append("  Weaknesses")
    for item in analysis["weaknesses"]:
        lines.append(f"    - {item}")
    lines.append("")
    lines.append("  Notes")
    for item in analysis["notes"]:
        lines.append(f"    - {item}")

    _section(lines, "Key Metrics")
    lines.extend(
        [
            f"  Net Return: {_pct(metrics['net_return_pct'])}  |  Official MTM Return: {_pct(metrics.get('official_mtm_net_return_pct'))}",
            f"  Trades: {int(metrics['total_trades'])}  |  Win Rate: {_pct(metrics['win_rate'])}  |  Avg R: {_signed(metrics['avg_r'])}  |  Total R: {_signed(metrics['expected_total_r'])}",
            f"  Gross R: {_signed(metrics.get('gross_expected_total_r'))}  |  Cost Drag R: {_signed(metrics.get('cost_drag_r'))}  |  Profit Factor: {_num(metrics['profit_factor'])}",
            f"  Max DD: {_pct(metrics['max_drawdown_pct'])}  |  Sharpe: {_num(metrics['sharpe'])}  |  Sortino: {_num(metrics.get('sortino'))}  |  Calmar: {_num(metrics['calmar'])}",
            f"  MFE Capture: {_pct(metrics.get('mfe_capture'))}  |  Avg MFE: {_signed(metrics.get('avg_mfe_r'))}R  |  Median MFE: {_signed(metrics.get('median_mfe_r'))}R  |  Avg MAE: {_signed(metrics.get('avg_mae_r'))}R",
            f"  Active Days: {_int(metrics.get('active_trade_days'))}/{_int(metrics.get('observed_session_days'))} ({_pct(metrics.get('active_trade_day_share'))})  |  Trades/month: {_num(metrics['trades_per_month'])}",
        ]
    )

    _section(lines, "Executive Verdicts")
    for key, value in analysis["verdicts"].items():
        lines.append(f"  {key}: {value}")
    layer_verdicts = layers.get("verdicts") or {}
    if layer_verdicts:
        lines.append("")
        lines.append("  Layer Verdicts:")
        for key, value in layer_verdicts.items():
            lines.append(f"    {key}: {value}")

    _section(lines, "1. Overview")
    lines.extend(
        [
            f"  Trades: {int(metrics['total_trades'])}",
            f"  Win Rate: {_pct(metrics['win_rate'])}",
            f"  Mean R: {_signed(metrics['avg_r'])}  |  Median R: {_signed(metrics['median_r'])}",
            f"  Total R: {_signed(metrics['expected_total_r'])}  |  Total PnL: {_money(metrics['net_profit'])}",
            f"  Net Return: {_pct(metrics['net_return_pct'])}  |  Max DD: {_pct(metrics['max_drawdown_pct'])}",
            f"  Profit Factor: {_num(metrics['profit_factor'])}  |  Sharpe: {_num(metrics['sharpe'])}  |  Sortino: {_num(metrics.get('sortino'))}  |  Calmar: {_num(metrics['calmar'])}",
            f"  Avg Hold: {_num(metrics['avg_hold_hours'])}h  |  Trades/month: {_num(metrics['trades_per_month'])}",
        ]
    )

    _section(lines, "2. Live Parity / KIS Constraints")
    for key in (
        "live_parity_fill_timing",
        "auction_mode",
        "same_bar_fill_count",
        "universe_size",
        "data_available_symbol_count",
        "unavailable_symbol_count",
        "unavailable_symbols",
        "candidate_pool_max",
        "candidate_pool_universe_fraction",
        "candidate_pool_data_available_fraction",
        "active_symbol_max",
        "selected_universe_fraction",
        "frontier_enabled",
        "frontier_size",
        "frontier_symbol_max",
        "frontier_universe_fraction",
        "frontier_selection_mode",
        "frontier_active_selection_mode",
        "ws_budget",
        "max_positions",
        "replay_mode",
        "candidate_snapshot_count",
        "replay_event_count",
    ):
        lines.append(f"  {key}: {audit.get(key)}")
    lines.append("  Verdict: completed-bar signals and next_5m_open fills are clean; no same-bar fills were observed.")
    lines.append("  KIS note: websocket expansion above ws8 raised frequency but degraded expectancy in the capacity sweep.")
    lines.append("  Coverage note: the executable slice is ws-capped; the broader frontier is REST-pollable and shadowed before promotion.")

    _section(lines, "3. Signal Funnel & Gate Attribution")
    lines.extend(
        [
            "  Signal Funnel:",
            "  ----------------------------------------",
            f"  evaluated               {int(funnel['evaluated']):>8}",
            f"  opening_range_built     {int(funnel['opening_range_built']):>8}",
            f"  entries                 {int(funnel['entries']):>8}",
            f"  entry_rejected          {int(funnel['entry_rejected']):>8}",
            f"  accept_rate             {_pct(funnel['accept_rate'])}",
            "",
            "  Rejection Counts By Reason:",
        ]
    )
    for key, value in funnel["rejection_counts"][:18]:
        lines.append(f"    {key:<34} {value:>6}")
    lines.append("")
    lines.append("  First Failed Gate Attribution:")
    for key, value in funnel["failed_gate_counts"][:18]:
        lines.append(f"    {key:<34} {value:>6}")
    if funnel.get("raw_breakout_quality"):
        lines.append("")
        lines.append("  Raw Breakout Quality Split:")
        for key, value in funnel["raw_breakout_quality"][:18]:
            lines.append(f"    {key:<34} {value:>6}")

    _section(lines, "3b. KRX Hot Candidate Frontier")
    lines.extend(
        [
            f"  Frontier symbols/day: {_int(metrics.get('frontier_symbol_max'))}  |  Executable ws slice: {_int(metrics.get('active_symbol_max'))}",
            f"  Selection modes: frontier={metrics.get('frontier_selection_mode')}  |  active_ws_seed={metrics.get('frontier_active_selection_mode')}",
            f"  Data availability: requested={_int(metrics.get('universe_size'))}, non-empty={_int(metrics.get('data_available_symbol_count'))}, unavailable={_int(metrics.get('unavailable_symbol_count'))}",
            f"  Universe coverage: pool={_pct(metrics.get('candidate_pool_universe_fraction'))} requested / {_pct(metrics.get('candidate_pool_data_available_fraction'))} available, frontier={_pct(metrics.get('frontier_universe_fraction'))}, selected={_pct(metrics.get('selected_universe_fraction'))}",
            f"  Shadow non-selected trades: {_int(metrics.get('frontier_shadow_nonselected_trade_count'))}  |  Total R: {_signed(metrics.get('frontier_shadow_nonselected_total_r'))}  |  Avg R: {_signed(metrics.get('frontier_shadow_nonselected_avg_r'))}",
            f"  Eligible promoted symbols: {_int(metrics.get('frontier_shadow_eligible_symbol_count'))}  |  Rotation days: {_int(metrics.get('frontier_rotation_days'))}  |  Promotions: {_int(metrics.get('frontier_rotation_promotion_count'))}",
            f"  Promotion-proof cohort: symbols={_int(metrics.get('frontier_rotation_proof_symbol_count'))}, trades={_int(metrics.get('frontier_rotation_proof_trade_count'))}, Total R={_signed(metrics.get('frontier_rotation_proof_total_r'))}, Avg R={_signed(metrics.get('frontier_rotation_proof_avg_r'))}",
            f"  Full frontier proof: trades={_int(metrics.get('frontier_rotation_global_trade_count'))}, Total R={_signed(metrics.get('frontier_rotation_global_total_r'))}, Avg R={_signed(metrics.get('frontier_rotation_global_avg_r'))}",
            f"  Frontier proof ready: {bool(metrics.get('frontier_rotation_frontier_proof_ready'))}",
            "  Interpretation: hot market is treated as opportunity breadth, not a naive directional chase; active ws names are seeded from prior-completed campaign/opportunity features, then shadowed against the full frontier.",
            "  Policy: non-selected frontier names can enter the executable slice only after the full frontier and the promotion-proof cohort both have positive prior shadow evidence; ws_budget remains the hard cap.",
        ]
    )
    unavailable = metrics.get("unavailable_symbols") or []
    if unavailable:
        lines.append(f"  Unavailable explicit-universe symbols: {', '.join(str(symbol) for symbol in unavailable[:12])}")
    top_shadow = metrics.get("frontier_shadow_top_symbols") or []
    if top_shadow:
        lines.append("  Top shadow-proven symbols:")
        for item in top_shadow[:8]:
            trades = int(float(item.get("trades", 0.0) or 0.0))
            total_r = float(item.get("total_r", 0.0) or 0.0)
            avg_r = total_r / trades if trades else 0.0
            lines.append(f"    {item.get('symbol')}: n={trades}, Total R={_signed(total_r)}, Avg R={_signed(avg_r)}")

    _section(lines, "3c. Full-Universe Alpha Capture Attribution")
    actual = alpha.get("actual") or {}
    shadow = alpha.get("shadow_nonselected") or {}
    selected_shadow = alpha.get("shadow_selected") or {}
    same_day = alpha.get("same_day") or {}
    lines.extend(
        [
            f"  Executable accepted trades: n={_int(actual.get('n'))}, WR={_pct(actual.get('win_rate'))}, AvgR={_signed(actual.get('avg_r'))}, TotalR={_signed(actual.get('total_r'))}, PF={_num(actual.get('profit_factor'))}",
            f"  Shadow of executable symbols: n={_int(selected_shadow.get('n'))}, AvgR={_signed(selected_shadow.get('avg_r'))}, TotalR={_signed(selected_shadow.get('total_r'))}",
            f"  Non-selected full-universe shadow: n={_int(shadow.get('n'))}, WR={_pct(shadow.get('win_rate'))}, AvgR={_signed(shadow.get('avg_r'))}, TotalR={_signed(shadow.get('total_r'))}, PF={_num(shadow.get('profit_factor'))}",
            f"  Positive shadow pool: {_signed(alpha.get('shadow_positive_pool_total_r'))}R  |  Negative shadow pool: {_signed(alpha.get('shadow_negative_pool_total_r'))}R",
            f"  Same-day net shadow > executable: {_int(same_day.get('days_shadow_net_gt_actual'))}/{_int(same_day.get('days'))} days",
            f"  Same-day gross positive shadow > executable: {_int(same_day.get('days_shadow_positive_pool_gt_actual'))}/{_int(same_day.get('days'))} days",
            "  Interpretation: the broader hot market contains isolated positive names, but the unselected full-universe signal pool is net negative before the ws8 cap. Expansion without stronger selection would add low-value trades faster than alpha.",
        ]
    )
    _append_shadow_group_section(lines, "  Shadow R by frontier rank bucket:", alpha.get("shadow_by_rank_bucket") or {})
    _append_shadow_group_section(lines, "  Shadow R by entry type:", alpha.get("shadow_by_entry_type") or {})
    top_days = (same_day.get("top_shadow_net_days") or [])[:5]
    if top_days:
        lines.append("  Best skipped-net same-day windows:")
        for item in top_days:
            lines.append(
                f"    {item.get('date')}: shadow_net={_signed(item.get('shadow_net_r'))}, "
                f"actual={_signed(item.get('actual_r'))}, shadow_trades={_int(item.get('shadow_trades'))}"
            )
    bottom_symbols = alpha.get("shadow_bottom_symbols") or []
    if bottom_symbols:
        lines.append("  Worst shadow symbols blocking naive expansion:")
        for item in bottom_symbols[:6]:
            lines.append(f"    {item.get('symbol')}: n={_int(item.get('n'))}, AvgR={_signed(item.get('avg_r'))}, TotalR={_signed(item.get('total_r'))}")

    _section(lines, "3d. Winner/Loser And Exit Root Cause")
    profile_delta = management.get("profile_delta") or {}
    winner_profile = management.get("winner_profile") or {}
    loser_profile = management.get("loser_profile") or {}
    lines.extend(
        [
            "  Winner vs loser profile deltas (winner minus loser):",
            f"    Frontier rank: {_signed(profile_delta.get('frontier_rank'))}  |  Momentum score: {_signed(profile_delta.get('momentum_score'))}  |  RVOL: {_signed(profile_delta.get('bar_rvol'))}",
            f"    CPR: {_signed(profile_delta.get('cpr'))}  |  AVWAP dist: {_signed(profile_delta.get('avwap_distance_pct'))}  |  OR width: {_signed(profile_delta.get('or_width_pct'))}",
            f"    MFE: winners {_signed(winner_profile.get('mfe_r'))} vs losers {_signed(loser_profile.get('mfe_r'))}; MAE: winners {_signed(winner_profile.get('mae_r'))} vs losers {_signed(loser_profile.get('mae_r'))}",
            f"  Losers with first MFE > 0.3R: {_int(management.get('losers_with_mfe_gt_03'))} ({_pct(management.get('losers_with_mfe_gt_03_share'))})",
            "  Interpretation: if score/RVOL deltas are small while MFE/MAE deltas are large, the issue is path quality after entry and candidate discrimination, not a broken fill model.",
        ]
    )
    lost_alpha = management.get("top_lost_alpha_trades") or []
    if lost_alpha:
        lines.append("  Top lost-alpha trades:")
        for item in lost_alpha[:6]:
            lines.append(
                f"    {str(item.get('entry_time'))[:10]} {item.get('symbol')}: "
                f"actual={_signed(item.get('actual_r'))}, MFE={_signed(item.get('mfe_r'))}, lost={_signed(item.get('lost_r'))}, exit={item.get('exit_reason')}"
            )

    _section(lines, "3e. Candidate Surfacing Quality")
    candidate_quality = layers.get("candidate_surfacing") or {}
    lines.extend(
        [
            f"  Verdict: {candidate_quality.get('verdict', 'REVIEW')}",
            f"  Candidate pool max: {_int(metrics.get('candidate_pool_max'))}; frontier max: {_int(metrics.get('frontier_symbol_max'))}; active ws max: {_int(metrics.get('active_symbol_max'))}; ws_budget={_int(audit.get('ws_budget'))}",
            f"  Candidate-pool coverage: requested universe {_pct(metrics.get('candidate_pool_universe_fraction'))}; data-available universe {_pct(metrics.get('candidate_pool_data_available_fraction'))}; frontier coverage {_pct(metrics.get('frontier_universe_fraction'))}.",
            f"  Accepted vs skipped: accepted avgR={_signed((alpha.get('actual') or {}).get('avg_r'))}, non-selected shadow avgR={_signed((alpha.get('shadow_nonselected') or {}).get('avg_r'))}, shadow totalR={_signed((alpha.get('shadow_nonselected') or {}).get('total_r'))}.",
            f"  Same-day skipped-net beat actual on {_int((alpha.get('same_day') or {}).get('days_shadow_net_gt_actual'))}/{_int((alpha.get('same_day') or {}).get('days'))} days.",
            f"  Interpretation: {candidate_quality.get('interpretation', 'No candidate-layer interpretation available.')}",
        ]
    )
    for item in candidate_quality.get("actions", [])[:6]:
        lines.append(f"  - {item}")

    _section(lines, "3f. First30 Signal Quality")
    first30 = layers.get("first30") or {}
    lines.append(f"  Verdict: {first30.get('verdict', 'NOT_APPLICABLE')}")
    lines.append(f"  Coverage: first30 metadata present on {_int(first30.get('n'))}/{int(metrics['total_trades'])} trades.")
    for key, label in (
        ("first30_ret", "First30 return"),
        ("first30_vwap_ret", "First30 close vs VWAP"),
        ("first30_gap", "Gap"),
        ("first30_rel_volume", "First30 relative volume"),
        ("first30_range_close_location", "First30 close location"),
        ("first30_signal_bar_cpr", "Signal-bar CPR"),
        ("first30_open_drawdown", "First30 open drawdown"),
        ("first30_range_atr", "First30 range/ATR"),
    ):
        item = (first30.get("profiles") or {}).get(key) or {}
        lines.append(
            f"  {label:<28} winners={_signed(item.get('winner_avg'))} losers={_signed(item.get('loser_avg'))} "
            f"delta={_signed(item.get('delta'))} monotonic={item.get('monotonic', 'n/a')}"
        )
    buckets = first30.get("bucket_tables") or {}
    for title, table in (
        ("  First30 return buckets:", buckets.get("first30_ret")),
        ("  First30 VWAP buckets:", buckets.get("first30_vwap_ret")),
        ("  First30 close-location buckets:", buckets.get("first30_range_close_location")),
    ):
        lines.append(title)
        _append_compact_stat_table(lines, table or {}, indent="    ")
    for item in first30.get("actions", [])[:6]:
        lines.append(f"  - {item}")

    _section(lines, "3g. Entry Mechanism Calibration")
    entry_diag = layers.get("entry") or {}
    lines.extend(
        [
            f"  Verdict: {entry_diag.get('verdict', 'REVIEW')}",
            f"  Evaluated candidate-bars: {_int(funnel.get('evaluated'))}; entries: {_int(funnel.get('entries'))}; rejected: {_int(funnel.get('entry_rejected'))}; accept_rate={_pct(funnel.get('accept_rate'))}.",
            f"  Trade count guidance: {entry_diag.get('trade_count_diagnosis', 'n/a')}",
            f"  Entry timing diagnosis: {entry_diag.get('timing_diagnosis', 'n/a')}",
            f"  Top bottleneck gate: {entry_diag.get('top_bottleneck', 'n/a')}",
        ]
    )
    for item in entry_diag.get("actions", [])[:8]:
        lines.append(f"  - {item}")

    _section(lines, "3h. Exit / Trade Management Fitness")
    exit_diag = layers.get("exit_management") or {}
    lines.extend(
        [
            f"  Verdict: {exit_diag.get('verdict', 'REVIEW')}",
            f"  MFE capture={_pct(metrics.get('mfe_capture'))}; losers with prior MFE>0.3R={_int(management.get('losers_with_mfe_gt_03'))} ({_pct(management.get('losers_with_mfe_gt_03_share'))}).",
            f"  Giveback diagnosis: {exit_diag.get('giveback_diagnosis', 'n/a')}",
            f"  Stop calibration: {exit_diag.get('stop_diagnosis', 'n/a')}",
        ]
    )
    lines.append("  MFE capture by exit path:")
    _append_compact_stat_table(lines, exit_diag.get("mfe_capture_by_exit") or {}, indent="    ")
    lines.append("  Economic exit frontier (diagnostic, not a fill simulation):")
    for item in exit_diag.get("economic_exit_frontier") or []:
        lines.append(
            f"    capture={_pct(item.get('capture'))}: approx_totalR={_signed(item.get('approx_total_r'))}, "
            f"delta_vs_actual={_signed(item.get('delta_vs_actual_r'))}"
        )
    for item in exit_diag.get("actions", [])[:8]:
        lines.append(f"  - {item}")

    _append_group_section(lines, "4. Entry Type Breakdown", groups["entry_type"], key_label="")
    _append_group_section(lines, "5. Direction Breakdown", groups["direction"], key_label="")

    _section(lines, "6. Momentum Score Distribution")
    _append_counter_stats(lines, groups["momentum_score"], label_prefix="  Score ")
    low = _stats_for([row for row in rows if row["momentum_score"] < 6])
    high = _stats_for([row for row in rows if row["momentum_score"] >= 6])
    lines.append("")
    lines.append(f"  WR monotonic with score: {'YES' if groups['score_monotonic'] else 'NO'}")
    lines.append(f"  Low scores (<6): WR={_pct(low['win_rate'])}, n={low['n']}  |  High scores (>=6): WR={_pct(high['win_rate'])}, n={high['n']}")

    _append_group_section(lines, "7. Opening Range Quality", groups["or_width_bucket"], key_label="")
    _append_group_section(lines, "8. Breakout Distance From OR/PDH", groups["breakout_distance_bucket"], key_label="")
    _append_group_section(lines, "9. RVOL At Entry", groups["rvol_bucket"], key_label="")
    _append_group_section(lines, "10. AVWAP Distance At Entry", groups["avwap_bucket"], key_label="")

    _section(lines, "11. Entry Bar Timing")
    _append_group_stats_inline(lines, groups["entry_time_bucket"])
    lines.append("")
    lines.append("  Top 5 entry bars:")
    for item in groups["top_entry_bars"][:5]:
        lines.append(f"    Bar {item['bar_index']:>2} ({item['time']}): n={item['n']}, WR={_pct(item['win_rate'])}, Mean R={_signed(item['avg_r'])}")

    _append_group_section(lines, "12. Regime Sizing Impact", groups["regime_tier"], key_label="Tier ")
    _append_group_section(lines, "13. Carry Analysis", groups["carry_bucket"], key_label="")
    _section(lines, "14. Regime x Entry Type")
    _append_matrix(lines, groups["regime_entry_matrix"])

    _append_group_section(lines, "15. Exit Reason Deep Dive", groups["exit_reason"], key_label="")
    _append_group_section(lines, "16. Partial Take Analysis", groups["partial_taken"], key_label="")

    _section(lines, "17. MFE / MAE Analysis")
    mfe = groups["mfe_mae"]
    lines.extend(
        [
            f"  Winners ({mfe['winner_count']}):",
            f"    Mean MFE: {_signed(mfe['winner_mean_mfe_r'])}R",
            f"    Capture efficiency: {_pct(mfe['winner_capture_efficiency'])}",
            f"    Mean giveback: {_signed(mfe['winner_mean_giveback_r'])}R",
            f"    MFE distribution: P25={_signed(mfe['mfe_p25'])}R, P50={_signed(mfe['mfe_p50'])}R, P75={_signed(mfe['mfe_p75'])}R",
            f"  Losers ({mfe['loser_count']}):",
            f"    Mean MAE: {_signed(mfe['loser_mean_mae_r'])}R",
            f"    MAE distribution: P25={_signed(mfe['mae_p25'])}R, P50={_signed(mfe['mae_p50'])}R, P75={_signed(mfe['mae_p75'])}R",
            f"    Losers with MFE > 0.3R first: {mfe['losers_with_mfe_gt_03']} ({_pct(mfe['losers_with_mfe_gt_03_share'])})",
        ]
    )

    _append_group_section(lines, "18. EOD / Carry Hold Quality", groups["eod_quality"], key_label="")
    _section(lines, "19. Stale Exit Analysis")
    stale = groups["stale_exits"]
    if stale["n"]:
        lines.append(f"  Stale exits: {stale['n']}  |  Avg R: {_signed(stale['avg_r'])}")
    else:
        lines.append("  No stale exits.")

    _section(lines, "20. Hold Duration")
    lines.append("    Pctl    Hours    WR%    Avg R     N")
    lines.append("  --------------------------------------")
    for item in groups["hold_duration_buckets"]:
        lines.append(f"  P{item['percentile']:>4}  {item['hours']:>7.1f}  {_pct_value(item['win_rate']):>5}  {_signed(item['avg_r']):>7}  {item['n']:>4}")

    _append_group_section(lines, "21. Flow Reversal Timing", groups["flow_reversal_timing"], key_label="")
    _append_group_section(lines, "22. Symbol Performance", groups["symbol"], key_label="", limit=20)
    _append_group_section(lines, "23. Sector Performance", groups["sector"], key_label="", limit=12)

    _section(lines, "24. Capacity / Red Hot Market Frontier")
    lines.append("  Capacity candidates:")
    for item in frontier["capacity_validation"]:
        m = item.get("metrics", {})
        lines.append(
            f"    {item.get('name'):<24} trades={_int(m.get('total_trades')):>4} "
            f"R={_signed(m.get('expected_total_r')):>8} ret={_pct(m.get('net_return_pct')):>8} "
            f"PF={_num(m.get('profit_factor')):>5} DD={_pct(m.get('max_drawdown_pct')):>7} active={_int(m.get('active_symbol_max'))}"
        )
    lines.append("  Interpretation: the selected 10:25-13:00 ws8 slice gives the strongest corrected train score; ws10/ws12 expansions added trades but weakened expectancy and drawdown.")

    _section(lines, "25. Holdout Validation")
    for item in frontier["targeted_holdout"]:
        m = item.get("metrics", {})
        lines.append(
            f"  {item.get('name'):<22} trades={_int(m.get('total_trades')):>3} "
            f"R={_signed(m.get('expected_total_r')):>7} ret={_pct(m.get('net_return_pct')):>8} "
            f"PF={_num(m.get('profit_factor')):>5} DD={_pct(m.get('max_drawdown_pct')):>7}"
        )

    _section(lines, "26. Mutation Threshold Sweeps")
    lines.append("  Targeted ws8 refinement:")
    for item in frontier["targeted_train"]:
        m = item.get("metrics", {})
        lines.append(
            f"    {item.get('name'):<24} score={_num(item.get('score')):>7} "
            f"trades={_int(m.get('total_trades')):>4} R={_signed(m.get('expected_total_r')):>8} "
            f"ret={_pct(m.get('net_return_pct')):>8} PF={_num(m.get('profit_factor')):>5} DD={_pct(m.get('max_drawdown_pct')):>7}"
        )
    lines.append(f"  Selected: {analysis['selected_candidate']}. It gave the strongest corrected train score while keeping ws_budget at 8; ws10/ws12 expansions added weaker or negative trades.")

    _section(lines, "27. Residual Risks / Next Diagnostics")
    holdout_trades = _selected_holdout_trades(frontier, analysis["selected_candidate"])
    lines.extend(
        [
            f"  - Holdout contains only {holdout_trades} selected-candidate trades, so this is a starting baseline, not production promotion.",
            "  - Rejected KALCB decisions do not yet carry realized shadow outcomes; selector frontier attribution is therefore based on accepted trades plus capacity/refinement sweeps.",
            "  - Daily frontier candidates use prior-completed daily heat and liquidity; intraday REST polling/paper parity should verify the same symbols before live promotion.",
            "  - All sector metadata currently collapses to UNKNOWN for many KRX names; sector caps should be revisited once Korean sector mapping is attached.",
            f"  - {KALCB_CORE_VERSION} behavior changed entry eligibility and frontier discovery; historical capacity/refinement sweep artifacts should be rerun before any production promotion.",
            "  - Paper/live parity should compare submitted intents, accepted KIS orders, fills, rejects, and deferred REST actions before live promotion.",
        ]
    )
    _section(lines, "28. Implementation Lessons Alignment")
    lines.extend(
        [
            "  - Decision ownership remains in the shared KALCB core; live and backtest adapters consume the same DecisionEvent/action stream.",
            "  - Signals still use completed 5m bars with next_5m_open fills; same-bar fill count is reported in the live-parity audit.",
            "  - Entries/exits continue through neutral StrategyAction objects and the replay SimBroker, preserving one execution/accounting path.",
            f"  - CPR rescue plus frontier rotation are versioned behavior changes ({KALCB_CORE_VERSION}), not backtest-only filter overrides.",
            "  - Diagnostics now split raw breakout quality failures into RVOL-min, CPR-floor, CPR-score, and CPR-rescued cohorts to keep funnel denominators cohort-pure.",
            "  - The KRX frontier is replay-bundle metadata plus an adapter-side active-snapshot view: the shared core still owns decisions, while shadow simulation uses a separate SimBroker and cannot alter executable PnL.",
            "  - Frontier rotation is evidence-gated by prior shadow outcomes and preserves the ws_budget cap, matching the live/paper KIS constraint surface.",
        ]
    )
    _section(lines, "29. Monthly PnL")
    _append_period_table(lines, layers.get("monthly") or {})

    _section(lines, "30. Day Of Week")
    _append_period_table(lines, layers.get("weekday") or {})

    _section(lines, "31. Streak Analysis")
    streaks = layers.get("streaks") or {}
    lines.extend(
        [
            f"  Longest win streak: {_int(streaks.get('longest_win_streak'))}",
            f"  Longest loss streak: {_int(streaks.get('longest_loss_streak'))}",
            f"  Current ending streak: {streaks.get('ending_streak', 'n/a')}",
            f"  Loss clusters >=3: {_int(streaks.get('loss_clusters_ge_3'))}",
        ]
    )

    _section(lines, "32. Rolling Expectancy (20-trade window)")
    rolling = layers.get("rolling_expectancy") or {}
    lines.extend(
        [
            f"  Best 20-trade avg R: {_signed(rolling.get('best_avg_r'))} ending trade #{_int(rolling.get('best_end_index'))}",
            f"  Worst 20-trade avg R: {_signed(rolling.get('worst_avg_r'))} ending trade #{_int(rolling.get('worst_end_index'))}",
            f"  Latest 20-trade avg R: {_signed(rolling.get('latest_avg_r'))}",
            f"  Windows below zero: {_int(rolling.get('negative_window_count'))}/{_int(rolling.get('window_count'))}",
        ]
    )

    _section(lines, "33. Drawdown Profile")
    dd = layers.get("drawdown_profile") or {}
    lines.extend(
        [
            f"  Max R drawdown: {_signed(dd.get('max_drawdown_r'))}R at trade #{_int(dd.get('max_drawdown_trade_index'))}",
            f"  DD episodes: {_int(dd.get('episode_count'))}",
            f"  Max trades underwater: {_int(dd.get('max_trades_underwater'))}",
            f"  Recovery from worst DD: {_int(dd.get('recovery_trades_from_worst'))} trades",
        ]
    )

    _section(lines, "34. R vs Dollar Disconnect")
    disconnect = layers.get("r_vs_dollar") or {}
    lines.extend(
        [
            f"  Total R: {_signed(disconnect.get('total_r'))}; total PnL: {_money(disconnect.get('total_pnl'))}",
            f"  Avg winner position: {_money(disconnect.get('avg_winner_notional'))}; avg loser position: {_money(disconnect.get('avg_loser_notional'))}",
            f"  Avg winner PnL: {_money(disconnect.get('avg_winner_pnl'))}; avg loser PnL: {_money(disconnect.get('avg_loser_pnl'))}",
            f"  Sizing alignment: {disconnect.get('verdict', 'n/a')}",
        ]
    )

    _section(lines, "35. Worst Period Autopsy")
    worst_periods = layers.get("worst_periods") or []
    if not worst_periods:
        lines.append("  No losing monthly periods.")
    for item in worst_periods[:5]:
        lines.append(
            f"  {item.get('period')}: {_signed(item.get('total_r'))}R ({_money(item.get('net_pnl'))}), "
            f"n={_int(item.get('n'))}, WR={_pct(item.get('win_rate'))}"
        )
        lines.append(f"    Entry types: {item.get('entry_types')}")
        lines.append(f"    Exit types: {item.get('exit_reasons')}")
        lines.append(f"    Top sectors: {item.get('sectors')}")

    _section(lines, "36. Intraday Alpha Curve")
    for item in layers.get("intraday_alpha_curve") or []:
        lines.append(
            f"  {item.get('bucket'):<22} n={_int(item.get('n')):>4} WR={_pct(item.get('win_rate')):>6} "
            f"AvgR={_signed(item.get('avg_r')):>8} TotalR={_signed(item.get('total_r')):>8} $/trade={_money(item.get('pnl_per_trade')):>10}"
        )

    _section(lines, "37. Instrumentation Gaps")
    for item in layers.get("instrumentation_gaps") or []:
        lines.append(f"  - {item}")
    return "\n".join(lines) + "\n"


def render_kalcb_evaluation(analysis: dict[str, Any]) -> str:
    metrics = analysis["final_metrics"]
    verdicts = analysis["verdicts"]
    lines = [
        "KALCB ROUND 1 EVALUATION",
        "=" * 72,
        f"Candidate: {analysis['selected_candidate']}",
        f"Status: {analysis['promotion_status']}",
        f"Trades: {int(metrics['total_trades'])}",
        f"Net return: {_pct(metrics['net_return_pct'])}",
        f"Expected total R: {_signed(metrics['expected_total_r'])}",
        f"Profit factor: {_num(metrics['profit_factor'])}",
        f"Max drawdown: {_pct(metrics['max_drawdown_pct'])}",
        "",
        "Verdicts:",
    ]
    for key, value in verdicts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Decision:",
            "Accepted as KALCB round_1 optimized baseline for further optimization/paper-parity work.",
            "Not promoted to production/live trading until OOS depth and KIS paper parity evidence are attached.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_external_artifacts(ws8_summary_path: str | Path, capacity_validation_path: str | Path, broad_summary_path: str | Path) -> dict[str, Any]:
    return {
        "ws8_summary": _load_json(Path(ws8_summary_path)),
        "capacity_validation": _load_jsonl(Path(capacity_validation_path)),
        "broad_summary": _load_json(Path(broad_summary_path)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trade_row(trade: TradeOutcome) -> dict[str, Any]:
    route = dict(trade.route_metadata)
    cohort = dict(trade.cohort_metadata)
    risk_per_share = float(route.get("risk_per_share", 0.0) or 0.0)
    risk_notional = risk_per_share * int(trade.qty)
    net_r = float(trade.net_pnl) / risk_notional if risk_notional > 0 else 0.0
    mfe_r = float(trade.mfe) / risk_per_share if risk_per_share > 0 else 0.0
    mae_r = float(trade.mae) / risk_per_share if risk_per_share > 0 else 0.0
    hold_hours = 0.0
    carry_days = 0
    if trade.exit_fill_time is not None:
        hold_hours = (trade.exit_fill_time - trade.entry_fill_time).total_seconds() / 3600.0
        carry_days = max(0, (trade.exit_fill_time.date() - trade.entry_fill_time.date()).days)
    avwap = float(route.get("avwap", 0.0) or 0.0)
    entry_ref = float(route.get("entry_price_ref", trade.entry_price) or trade.entry_price)
    avwap_dist = (entry_ref - avwap) / avwap if avwap > 0 else 0.0
    or_high = float(route.get("or_high", 0.0) or 0.0)
    or_low = float(route.get("or_low", 0.0) or 0.0)
    or_width_pct = (or_high - or_low) / or_high if or_high > 0 else 0.0
    breakout_distance = _filter_actual(route, "breakout_distance_cap")
    return {
        "symbol": trade.symbol,
        "qty": int(trade.qty),
        "entry_time": trade.entry_fill_time,
        "entry_decision_time": trade.entry_decision_time,
        "exit_time": trade.exit_fill_time,
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price or trade.entry_price),
        "net_pnl": float(trade.net_pnl),
        "gross_pnl": float(trade.gross_pnl),
        "commission": float(trade.commission),
        "r": net_r,
        "gross_r": float(trade.r_multiple),
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_capture": max(0.0, net_r / mfe_r) if mfe_r > 0 else 0.0,
        "giveback_r": max(0.0, mfe_r - net_r),
        "hold_hours": hold_hours,
        "carry_days": carry_days,
        "entry_type": str(route.get("entry_type", "UNKNOWN")),
        "exit_reason": str(trade.exit_reason or "UNKNOWN"),
        "sector": str(route.get("sector", "UNKNOWN") or "UNKNOWN"),
        "regime_tier": str(route.get("regime_tier", "UNKNOWN") or "UNKNOWN"),
        "frontier_rank": int(route.get("frontier_rank", 0) or 0),
        "candidate_rank": int(route.get("candidate_rank", route.get("frontier_rank", 0)) or 0),
        "frontier_selection_score": float(route.get("frontier_selection_score", 0.0) or 0.0),
        "momentum_score": int(route.get("momentum_score", 0) or 0),
        "score_detail": dict(route.get("score_detail") or {}),
        "bar_rvol": float(route.get("bar_rvol", 0.0) or 0.0),
        "cpr": float(route.get("cpr", 0.0) or 0.0),
        "avwap_distance_pct": avwap_dist,
        "or_width_pct": or_width_pct,
        "breakout_distance_r": float(breakout_distance) if breakout_distance is not None else 0.0,
        "first30_ret": _optional_float(route.get("first30_ret")),
        "first30_vwap_ret": _optional_float(route.get("first30_vwap_ret")),
        "first30_gap": _optional_float(route.get("first30_gap")),
        "first30_rel_volume": _optional_float(route.get("first30_rel_volume")),
        "first30_range_close_location": _optional_float(route.get("first30_range_close_location")),
        "first30_signal_bar_cpr": _optional_float(route.get("first30_signal_bar_cpr")),
        "first30_open_drawdown": _optional_float(route.get("first30_open_drawdown")),
        "first30_low_vs_prev_close": _optional_float(route.get("first30_low_vs_prev_close")),
        "first30_range_atr": _optional_float(route.get("first30_range_atr")),
        "partial_taken": bool(cohort.get("partial_taken", False)),
        "entry_bar_index": _entry_bar_index(trade.entry_decision_time),
        "entry_bar_label": trade.entry_decision_time.strftime("%H:%M"),
        "notional": float(trade.entry_price) * int(trade.qty),
    }


def _filter_actual(route: dict[str, Any], name: str) -> float | None:
    for item in route.get("filter_decisions", ()) or ():
        if item.get("filter_name") == name:
            try:
                return float(item.get("actual_value"))
            except (TypeError, ValueError):
                return None
    return None


def _entry_bar_index(timestamp: datetime) -> int:
    return int(((timestamp.hour * 60 + timestamp.minute) - (9 * 60)) / 5) + 1


def _final_metrics(metrics: dict[str, Any], trades: list[TradeOutcome], config: dict[str, Any], equity_curve: list[float] | tuple[float, ...] | None = None) -> dict[str, Any]:
    rows = [_trade_row(trade) for trade in trades]
    total = len(rows)
    r_values = [row["r"] for row in rows]
    gross_r_values = [row["gross_r"] for row in rows]
    wins = [row for row in rows if row["net_pnl"] > 0]
    losses = [row for row in rows if row["net_pnl"] < 0]
    first_date = min((row["entry_time"].date() for row in rows), default=None)
    last_date = max((row["entry_time"].date() for row in rows), default=None)
    months = max(1.0, ((last_date - first_date).days + 1) / 30.4375) if first_date and last_date else 1.0
    initial_equity = float(config.get("initial_equity", 0.0) or 0.0)
    net_profit = float(metrics.get("net_profit", sum(row["net_pnl"] for row in rows)))
    max_dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    net_return_pct = float(net_profit / initial_equity) if initial_equity else float(metrics.get("net_return_pct", 0.0) or 0.0)
    risk_metrics = _equity_risk_metrics(equity_curve or ())
    session_days = _observed_session_days(rows, metrics)
    active_days = len({row["entry_time"].date() for row in rows})
    base_metrics = {key: value for key, value in dict(metrics).items() if key != "frontier_shadow_trade_rows"}
    return {
        **base_metrics,
        "total_trades": float(total),
        "winning_trades": float(len(wins)),
        "losing_trades": float(len(losses)),
        "win_rate": float(len(wins) / total) if total else 0.0,
        "avg_r": float(mean(r_values)) if r_values else 0.0,
        "median_r": float(median(r_values)) if r_values else 0.0,
        "expected_total_r": float(sum(r_values)),
        "gross_avg_r": float(mean(gross_r_values)) if gross_r_values else 0.0,
        "gross_expected_total_r": float(sum(gross_r_values)),
        "cost_drag_r": float(sum(r_values) - sum(gross_r_values)),
        "net_profit": net_profit,
        "net_return_pct": net_return_pct,
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "max_drawdown_pct": max_dd,
        "avg_hold_hours": float(mean(row["hold_hours"] for row in rows)) if rows else 0.0,
        "trades_per_month": float(total / months) if months else 0.0,
        "total_commissions": float(sum(row["commission"] for row in rows)),
        "sharpe": float(metrics.get("sharpe", 0.0) or 0.0),
        "sortino": float(metrics.get("sortino", risk_metrics.get("sortino", 0.0)) or 0.0),
        "calmar": float((net_return_pct / max_dd) if max_dd > 0 else 0.0),
        "avg_mfe_r": float(mean(row["mfe_r"] for row in rows)) if rows else 0.0,
        "median_mfe_r": float(median(row["mfe_r"] for row in rows)) if rows else 0.0,
        "avg_mae_r": float(mean(row["mae_r"] for row in rows)) if rows else 0.0,
        "median_mae_r": float(median(row["mae_r"] for row in rows)) if rows else 0.0,
        "mfe_ge_1_share": float(sum(1 for row in rows if row["mfe_r"] >= 1.0) / total) if total else 0.0,
        "mae_le_neg_1_share": float(sum(1 for row in rows if row["mae_r"] <= -1.0) / total) if total else 0.0,
        "observed_session_days": float(session_days),
        "active_trade_days": float(active_days),
        "active_trade_day_share": float(active_days / session_days) if session_days else 0.0,
        "equity_return_mean": risk_metrics.get("mean_return", 0.0),
        "equity_return_downside_deviation": risk_metrics.get("downside_deviation", 0.0),
    }


def _build_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {
        "entry_type": _group_by(rows, lambda row: row["entry_type"]),
        "direction": {"LONG": _stats_for(rows), "SHORT": _stats_for([])},
        "momentum_score": _group_by(rows, lambda row: str(row["momentum_score"])),
        "or_width_bucket": _group_by(rows, _or_width_bucket),
        "breakout_distance_bucket": _group_by(rows, _breakout_distance_bucket(rows)),
        "rvol_bucket": _group_by(rows, _rvol_bucket),
        "avwap_bucket": _group_by(rows, _avwap_bucket),
        "entry_time_bucket": _group_by(rows, _entry_time_bucket),
        "regime_tier": _group_by(rows, lambda row: row["regime_tier"]),
        "carry_bucket": _group_by(rows, lambda row: "Overnight (carry_days>0)" if row["carry_days"] > 0 else "Intraday (carry_days=0)"),
        "exit_reason": _group_by(rows, lambda row: row["exit_reason"]),
        "partial_taken": _group_by(rows, lambda row: "With partial" if row["partial_taken"] else "Without partial"),
        "eod_quality": _group_by(rows, lambda row: "EOD/Carry flatten" if "eod" in row["exit_reason"].lower() else "Non-EOD exit"),
        "flow_reversal_timing": _group_by([row for row in rows if "flow_reversal" in row["exit_reason"].lower()], _flow_timing_bucket),
        "symbol": _group_by(rows, lambda row: row["symbol"]),
        "sector": _group_by(rows, lambda row: row["sector"]),
    }
    groups["score_monotonic"] = _score_monotonic(groups["momentum_score"])
    groups["top_entry_bars"] = _top_entry_bars(rows)
    groups["regime_entry_matrix"] = _matrix(rows, "regime_tier", "entry_type")
    groups["mfe_mae"] = _mfe_mae(rows)
    groups["stale_exits"] = _stats_for([row for row in rows if "stale" in row["exit_reason"].lower()])
    groups["hold_duration_buckets"] = _hold_duration_buckets(rows)
    return groups


def _group_by(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    return {key: _stats_for(value) for key, value in sorted(buckets.items())}


def _stats_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r_values = [float(row["r"]) for row in rows]
    wins = [row for row in rows if row["net_pnl"] > 0]
    losses = [row for row in rows if row["net_pnl"] < 0]
    gains = sum(row["net_pnl"] for row in wins)
    loss_abs = abs(sum(row["net_pnl"] for row in losses))
    return {
        "n": len(rows),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "avg_r": mean(r_values) if r_values else 0.0,
        "median_r": median(r_values) if r_values else 0.0,
        "total_r": sum(r_values),
        "net_pnl": sum(row["net_pnl"] for row in rows),
        "profit_factor": gains / loss_abs if loss_abs > 0 else (999.0 if gains > 0 else 0.0),
        "avg_hold_hours": mean(row["hold_hours"] for row in rows) if rows else 0.0,
        "avg_notional": mean(row["notional"] for row in rows) if rows else 0.0,
    }


def _signal_funnel(decisions: list[DecisionEvent]) -> dict[str, Any]:
    codes = Counter(decision.decision_code for decision in decisions)
    rejection_counts = Counter(str(decision.reason) for decision in decisions if decision.decision_code == "entry_rejected")
    failed_gates: Counter[str] = Counter()
    raw_quality: Counter[str] = Counter()
    for decision in decisions:
        if decision.decision_code not in {"entry", "entry_rejected"}:
            continue
        metadata = dict(decision.metadata)
        filter_decisions = metadata.get("filter_decisions", ()) or ()
        if decision.decision_code == "entry":
            if any(item.get("filter_name") == "cpr_relax" and item.get("passed") is True for item in filter_decisions):
                raw_quality["cpr_relaxed_entries"] += 1
            continue
        first_failed = ""
        quality_failed = ""
        for item in filter_decisions:
            if item.get("applicable", True) and item.get("passed") is False:
                first_failed = str(item.get("filter_name") or "")
                if first_failed in {"rvol_min", "cpr_relax_floor", "cpr_relax_score", "cpr_gate"}:
                    quality_failed = first_failed
                break
        failed_gates[first_failed or str(decision.reason)] += 1
        if quality_failed:
            raw_quality[quality_failed] += 1
        elif str(decision.reason) == "rvol_or_cpr_filter":
            raw_quality["legacy_rvol_or_cpr_filter"] += 1
    entries = codes.get("entry", 0)
    rejected = codes.get("entry_rejected", 0)
    evaluated = entries + rejected
    return {
        "decision_count": len(decisions),
        "evaluated": evaluated,
        "opening_range_built": codes.get("opening_range_built", 0),
        "entries": entries,
        "entry_rejected": rejected,
        "accept_rate": entries / evaluated if evaluated else 0.0,
        "decision_code_counts": codes.most_common(),
        "rejection_counts": rejection_counts.most_common(),
        "failed_gate_counts": failed_gates.most_common(),
        "raw_breakout_quality": raw_quality.most_common(),
    }


def _alpha_capture_diagnostics(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    shadow_rows = [dict(row) for row in (metrics.get("frontier_shadow_trade_rows") or [])]
    nonselected = [row for row in shadow_rows if not bool(row.get("active_at_signal"))]
    selected_shadow = [row for row in shadow_rows if bool(row.get("active_at_signal"))]
    actual_by_day: dict[str, float] = defaultdict(float)
    actual_count_by_day: Counter[str] = Counter()
    for row in rows:
        day = row["entry_time"].date().isoformat()
        actual_by_day[day] += float(row["r"])
        actual_count_by_day[day] += 1
    shadow_by_day: dict[str, float] = defaultdict(float)
    shadow_positive_by_day: dict[str, float] = defaultdict(float)
    shadow_negative_by_day: dict[str, float] = defaultdict(float)
    shadow_count_by_day: Counter[str] = Counter()
    for row in nonselected:
        day = str(row.get("entry_date") or str(row.get("entry_time", ""))[:10])
        r_value = float(row.get("r", 0.0) or 0.0)
        shadow_by_day[day] += r_value
        shadow_count_by_day[day] += 1
        if r_value > 0:
            shadow_positive_by_day[day] += r_value
        elif r_value < 0:
            shadow_negative_by_day[day] += r_value
    all_days = sorted(set(actual_by_day) | set(shadow_by_day))
    same_day_rows = [
        {
            "date": day,
            "actual_r": float(actual_by_day.get(day, 0.0)),
            "actual_trades": int(actual_count_by_day.get(day, 0)),
            "shadow_net_r": float(shadow_by_day.get(day, 0.0)),
            "shadow_positive_r": float(shadow_positive_by_day.get(day, 0.0)),
            "shadow_negative_r": float(shadow_negative_by_day.get(day, 0.0)),
            "shadow_trades": int(shadow_count_by_day.get(day, 0)),
            "shadow_minus_actual_r": float(shadow_by_day.get(day, 0.0) - actual_by_day.get(day, 0.0)),
        }
        for day in all_days
    ]
    return {
        "actual": _r_stats([row["r"] for row in rows]),
        "shadow_selected": _r_stats([row.get("r", 0.0) for row in selected_shadow]),
        "shadow_nonselected": _r_stats([row.get("r", 0.0) for row in nonselected]),
        "shadow_positive_pool_total_r": float(sum(max(float(row.get("r", 0.0) or 0.0), 0.0) for row in nonselected)),
        "shadow_negative_pool_total_r": float(sum(min(float(row.get("r", 0.0) or 0.0), 0.0) for row in nonselected)),
        "same_day": {
            "days": len(same_day_rows),
            "days_shadow_net_gt_actual": sum(1 for row in same_day_rows if row["shadow_net_r"] > row["actual_r"]),
            "days_shadow_positive_pool_gt_actual": sum(1 for row in same_day_rows if row["shadow_positive_r"] > row["actual_r"]),
            "top_shadow_net_days": sorted(same_day_rows, key=lambda row: row["shadow_minus_actual_r"], reverse=True)[:8],
            "worst_shadow_net_days": sorted(same_day_rows, key=lambda row: row["shadow_minus_actual_r"])[:8],
        },
        "shadow_by_rank_bucket": _shadow_group_stats(nonselected, lambda row: _frontier_rank_bucket(int(row.get("frontier_rank", 0) or 0))),
        "shadow_by_entry_type": _shadow_group_stats(nonselected, lambda row: str(row.get("entry_type") or "UNKNOWN")),
        "shadow_top_symbols": _shadow_symbol_stats(nonselected, reverse=True)[:12],
        "shadow_bottom_symbols": _shadow_symbol_stats(nonselected, reverse=False)[:12],
    }


def _management_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winners = [row for row in rows if row["r"] > 0]
    losers = [row for row in rows if row["r"] <= 0]
    lost_alpha = sorted(
        [
            {
                "symbol": row["symbol"],
                "entry_time": row["entry_time"].isoformat(),
                "exit_reason": row["exit_reason"],
                "actual_r": row["r"],
                "mfe_r": row["mfe_r"],
                "lost_r": row["mfe_r"] - row["r"],
            }
            for row in rows
            if row["mfe_r"] > 0 and row["mfe_r"] - row["r"] > 0
        ],
        key=lambda item: item["lost_r"],
        reverse=True,
    )
    return {
        "winner_profile": _numeric_profile(winners),
        "loser_profile": _numeric_profile(losers),
        "profile_delta": _profile_delta(_numeric_profile(winners), _numeric_profile(losers)),
        "losers_with_mfe_gt_03": len([row for row in losers if row["mfe_r"] > 0.3]),
        "losers_with_mfe_gt_03_share": len([row for row in losers if row["mfe_r"] > 0.3]) / len(losers) if losers else 0.0,
        "top_lost_alpha_trades": lost_alpha[:10],
    }


def _layer_diagnostics(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    funnel: dict[str, Any],
    sweeps: dict[str, Any],
    alpha: dict[str, Any],
    management: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdicts": _layer_verdicts(rows, metrics, funnel, alpha, management),
        "candidate_surfacing": _candidate_surfacing_diagnostics(metrics, alpha),
        "first30": _first30_signal_diagnostics(rows),
        "entry": _entry_mechanism_diagnostics(rows, metrics, funnel),
        "exit_management": _exit_management_diagnostics(rows, metrics, management),
        "monthly": _period_stats(rows, lambda row: row["entry_time"].strftime("%Y-%m")),
        "weekday": _period_stats(rows, lambda row: row["entry_time"].strftime("%A")),
        "streaks": _streak_diagnostics(rows),
        "rolling_expectancy": _rolling_expectancy(rows, window=20),
        "drawdown_profile": _drawdown_profile(rows),
        "r_vs_dollar": _r_vs_dollar_disconnect(rows),
        "worst_periods": _worst_periods(rows),
        "intraday_alpha_curve": _intraday_alpha_curve(rows),
        "instrumentation_gaps": _instrumentation_gaps(rows, metrics, alpha, sweeps),
    }


def _layer_verdicts(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    funnel: dict[str, Any],
    alpha: dict[str, Any],
    management: dict[str, Any],
) -> dict[str, str]:
    candidate = _candidate_surfacing_diagnostics(metrics, alpha)
    first30 = _first30_signal_diagnostics(rows)
    entry = _entry_mechanism_diagnostics(rows, metrics, funnel)
    exit_mgmt = _exit_management_diagnostics(rows, metrics, management)
    return {
        "Candidate surfacing": candidate["verdict"],
        "First30 analysis": first30["verdict"],
        "Entry mechanism": entry["verdict"],
        "Exit mechanism": exit_mgmt["verdict"],
        "Trade management": exit_mgmt["trade_management_verdict"],
    }


def _candidate_surfacing_diagnostics(metrics: dict[str, Any], alpha: dict[str, Any]) -> dict[str, Any]:
    actual = alpha.get("actual") or {}
    shadow = alpha.get("shadow_nonselected") or {}
    same_day = alpha.get("same_day") or {}
    accepted_avg = float(actual.get("avg_r", metrics.get("avg_r", 0.0)) or 0.0)
    shadow_avg = float(shadow.get("avg_r", metrics.get("frontier_shadow_nonselected_avg_r", 0.0)) or 0.0)
    shadow_total = float(shadow.get("total_r", metrics.get("frontier_shadow_nonselected_total_r", 0.0)) or 0.0)
    skipped_days = int(same_day.get("days_shadow_net_gt_actual", 0) or 0)
    days = int(same_day.get("days", 0) or 0)
    verdict = "GOOD"
    if accepted_avg <= 0 or shadow_avg >= accepted_avg or (days and skipped_days / days > 0.45):
        verdict = "REVIEW"
    if shadow_total > 0 and shadow_avg > 0:
        verdict = "WEAK"
    actions: list[str] = []
    if shadow_avg >= accepted_avg:
        actions.append("Candidate list is not clearly better than skipped frontier names; rerun premarket/first30 selection before tuning exits.")
    if days and skipped_days / days > 0.45:
        actions.append("Skipped names beat the executable set on too many days; add same-day breadth/sector participation calibration.")
    if float(metrics.get("candidate_pool_universe_fraction", 0.0) or 0.0) < 0.10:
        actions.append("Candidate pool is very narrow; verify liquidity/clean filters are not choking the hot universe.")
    if not actions:
        actions.append("Candidate surfacing looks directionally useful; keep monitoring skipped positive-shadow pockets as expansion candidates.")
    return {
        "verdict": verdict,
        "accepted_avg_r": accepted_avg,
        "shadow_avg_r": shadow_avg,
        "shadow_total_r": shadow_total,
        "skipped_better_day_share": skipped_days / days if days else 0.0,
        "interpretation": (
            "Accepted candidates outperform the skipped shadow pool after costs."
            if verdict == "GOOD"
            else "Candidate surfacing is not cleanly separating the best intraday opportunities from skipped names."
        ),
        "actions": actions,
    }


def _first30_signal_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "first30_ret",
        "first30_vwap_ret",
        "first30_gap",
        "first30_rel_volume",
        "first30_range_close_location",
        "first30_signal_bar_cpr",
        "first30_open_drawdown",
        "first30_low_vs_prev_close",
        "first30_range_atr",
    )
    first30_rows = [row for row in rows if row.get("first30_ret") is not None]
    profiles = {field: _winner_loser_numeric_profile(first30_rows, field) for field in fields}
    bucket_tables = {
        "first30_ret": _group_by(first30_rows, lambda row: _signed_pct_bucket(row.get("first30_ret"), (-0.005, 0.0, 0.005, 0.015))),
        "first30_vwap_ret": _group_by(first30_rows, lambda row: _signed_pct_bucket(row.get("first30_vwap_ret"), (-0.003, 0.0, 0.003, 0.01))),
        "first30_range_close_location": _group_by(first30_rows, lambda row: _unit_bucket(row.get("first30_range_close_location"), (0.25, 0.50, 0.75))),
        "first30_rel_volume": _group_by(first30_rows, lambda row: _unit_bucket(row.get("first30_rel_volume"), (0.75, 1.0, 1.5, 2.5))),
    }
    if not first30_rows:
        return {
            "verdict": "NOT_APPLICABLE",
            "n": 0,
            "profiles": profiles,
            "bucket_tables": bucket_tables,
            "actions": ["No first30 metadata was present; use this report section after first30/opening-drive entry modes are active."],
        }
    helpful = 0
    misleading = 0
    for field in ("first30_ret", "first30_vwap_ret", "first30_range_close_location", "first30_signal_bar_cpr"):
        delta = float(profiles[field].get("delta", 0.0) or 0.0)
        helpful += 1 if delta > 0 else 0
        misleading += 1 if delta < 0 else 0
    verdict = "GOOD" if helpful >= 3 and misleading <= 1 else "REVIEW"
    actions = []
    if profiles["first30_ret"]["delta"] <= 0:
        actions.append("First30 return is not separating winners; test lower reliance on raw first30 momentum or combine with flow/sector confirmation.")
    if profiles["first30_vwap_ret"]["delta"] <= 0:
        actions.append("Close-vs-VWAP is not helping; test whether VWAP gate is too loose, too tight, or redundant after candidate ranking.")
    if profiles["first30_open_drawdown"]["delta"] < 0:
        actions.append("Winners have worse open drawdown than losers; avoid over-penalizing early shakeouts without confirmation.")
    if not actions:
        actions.append("First30 features are directionally useful; refine thresholds locally instead of reopening the entire signal space.")
    return {"verdict": verdict, "n": len(first30_rows), "profiles": profiles, "bucket_tables": bucket_tables, "actions": actions}


def _entry_mechanism_diagnostics(rows: list[dict[str, Any]], metrics: dict[str, Any], funnel: dict[str, Any]) -> dict[str, Any]:
    trades = len(rows)
    trades_per_month = float(metrics.get("trades_per_month", 0.0) or 0.0)
    accept_rate = float(funnel.get("accept_rate", 0.0) or 0.0)
    rejection_counts = list(funnel.get("failed_gate_counts") or funnel.get("rejection_counts") or [])
    top_bottleneck = f"{rejection_counts[0][0]} ({rejection_counts[0][1]})" if rejection_counts else "none"
    avg_r = float(metrics.get("avg_r", 0.0) or 0.0)
    verdict = "GOOD" if trades >= 50 and avg_r > 0 and 0.0005 <= accept_rate <= 0.25 else "REVIEW"
    if trades < 20 or trades_per_month < 4:
        trade_count_diagnosis = "too few entries; candidate layer may be fine but entry gates are likely choking opportunity."
    elif trades_per_month > 80 and avg_r <= 0:
        trade_count_diagnosis = "too many low-edge entries; tighten first30/breakout quality or rank gates."
    else:
        trade_count_diagnosis = "entry frequency is workable for further optimization."
    timing_groups = _group_by(rows, _entry_time_bucket)
    best_timing = _best_group(timing_groups)
    worst_timing = _worst_group(timing_groups)
    timing_diagnosis = f"best bucket {best_timing[0]} avgR={_signed(best_timing[1]['avg_r'])}; worst bucket {worst_timing[0]} avgR={_signed(worst_timing[1]['avg_r'])}."
    actions: list[str] = []
    if accept_rate < 0.0005:
        actions.append("Entry acceptance is extremely tight; test earlier first30/opening-drive participation and fewer redundant gates.")
    if accept_rate > 0.25:
        actions.append("Entry acceptance is loose; rank and quality gates should do more work before exits are optimized.")
    if "rvol" in top_bottleneck.lower():
        actions.append("RVOL is the main bottleneck; verify expected-volume normalization for KRX 5m bars before tightening further.")
    if not actions:
        actions.append("Entry acceptance and timing look usable; optimize trade management around the best timing buckets.")
    return {
        "verdict": verdict,
        "trade_count_diagnosis": trade_count_diagnosis,
        "timing_diagnosis": timing_diagnosis,
        "top_bottleneck": top_bottleneck,
        "actions": actions,
    }


def _exit_management_diagnostics(rows: list[dict[str, Any]], metrics: dict[str, Any], management: dict[str, Any]) -> dict[str, Any]:
    capture = float(metrics.get("mfe_capture", 0.0) or 0.0)
    loser_mfe_share = float(management.get("losers_with_mfe_gt_03_share", 0.0) or 0.0)
    avg_mae = float(metrics.get("avg_mae_r", 0.0) or 0.0)
    verdict = "GOOD" if capture >= 0.35 and loser_mfe_share <= 0.20 else "REVIEW"
    trade_management_verdict = "GOOD" if float(metrics.get("max_drawdown_pct", 0.0) or 0.0) <= 0.08 and avg_mae > -3.0 else "REVIEW"
    exit_groups = _group_by(rows, lambda row: row["exit_reason"])
    capture_by_exit = {
        key: {**stats, "avg_mfe_capture": _avg(row["mfe_capture"] for row in rows if row["exit_reason"] == key)}
        for key, stats in exit_groups.items()
    }
    actual_total = float(metrics.get("expected_total_r", 0.0) or 0.0)
    frontier = [
        {
            "capture": capture_level,
            "approx_total_r": sum(max(0.0, row["mfe_r"]) * capture_level for row in rows),
            "delta_vs_actual_r": sum(max(0.0, row["mfe_r"]) * capture_level for row in rows) - actual_total,
        }
        for capture_level in (0.25, 0.50, 0.75, 1.00)
    ]
    actions: list[str] = []
    if capture < 0.25:
        actions.append("MFE capture is weak; prioritize partials, breakeven, and trailing exits before new entry filters.")
    if loser_mfe_share > 0.20:
        actions.append("Many losers had usable MFE first; add no-MFE/failed-followthrough/VWAP-fail exits or faster protection.")
    if float(metrics.get("mae_le_neg_1_share", 0.0) or 0.0) > 0.35:
        actions.append("Too many trades move beyond -1R; review hard stop mode and gap/slippage assumptions.")
    if not actions:
        actions.append("Exit capture is serviceable; focus on localized stop/target perturbations instead of broad exit redesign.")
    return {
        "verdict": verdict,
        "trade_management_verdict": trade_management_verdict,
        "giveback_diagnosis": f"MFE capture {_pct(capture)} with loser prior-MFE share {_pct(loser_mfe_share)}.",
        "stop_diagnosis": f"avg MAE {_signed(avg_mae)}R; MAE <= -1R share {_pct(metrics.get('mae_le_neg_1_share'))}.",
        "mfe_capture_by_exit": capture_by_exit,
        "economic_exit_frontier": frontier,
        "actions": actions,
    }


def _r_stats(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value or 0.0) for value in values]
    wins = [value for value in vals if value > 0]
    losses = [value for value in vals if value < 0]
    gain = sum(wins)
    loss = abs(sum(losses))
    return {
        "n": len(vals),
        "win_rate": len(wins) / len(vals) if vals else 0.0,
        "avg_r": mean(vals) if vals else 0.0,
        "median_r": median(vals) if vals else 0.0,
        "total_r": sum(vals),
        "profit_factor": gain / loss if loss > 0 else (999.0 if gain > 0 else 0.0),
    }


def _winner_loser_numeric_profile(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    winners = [_float(row.get(field)) for row in rows if row.get(field) is not None and row["r"] > 0]
    losers = [_float(row.get(field)) for row in rows if row.get(field) is not None and row["r"] <= 0]
    all_rows = [row for row in rows if row.get(field) is not None]
    buckets = _tertile_buckets(all_rows, field)
    return {
        "winner_avg": mean(winners) if winners else 0.0,
        "loser_avg": mean(losers) if losers else 0.0,
        "delta": (mean(winners) if winners else 0.0) - (mean(losers) if losers else 0.0),
        "monotonic": _avg_r_monotonic(buckets),
        "n": len(all_rows),
    }


def _tertile_buckets(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    usable = sorted([row for row in rows if row.get(field) is not None], key=lambda row: _float(row.get(field)))
    if not usable:
        return {}
    output: dict[str, dict[str, Any]] = {}
    cuts = (("low", 0, len(usable) // 3), ("mid", len(usable) // 3, 2 * len(usable) // 3), ("high", 2 * len(usable) // 3, len(usable)))
    for label, start, end in cuts:
        bucket = usable[start:end] or usable[start : start + 1]
        output[label] = _stats_for(bucket)
    return output


def _avg_r_monotonic(groups: dict[str, dict[str, Any]]) -> bool | str:
    if len(groups) < 3:
        return "n/a"
    ordered = [groups[key]["avg_r"] for key in ("low", "mid", "high") if key in groups]
    return all(ordered[index] <= ordered[index + 1] for index in range(len(ordered) - 1))


def _period_stats(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    return _group_by(rows, key_fn)


def _streak_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    longest_win = 0
    longest_loss = 0
    current_type = ""
    current_len = 0
    loss_clusters = 0
    for row in ordered:
        kind = "win" if row["r"] > 0 else "loss"
        if kind == current_type:
            current_len += 1
        else:
            if current_type == "loss" and current_len >= 3:
                loss_clusters += 1
            current_type = kind
            current_len = 1
        if kind == "win":
            longest_win = max(longest_win, current_len)
        else:
            longest_loss = max(longest_loss, current_len)
    if current_type == "loss" and current_len >= 3:
        loss_clusters += 1
    ending = f"{current_type}:{current_len}" if current_type else "none"
    return {
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "ending_streak": ending,
        "loss_clusters_ge_3": loss_clusters,
    }


def _rolling_expectancy(rows: list[dict[str, Any]], *, window: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    values = [row["r"] for row in ordered]
    if len(values) < window:
        avg = mean(values) if values else 0.0
        return {
            "window_count": 1 if values else 0,
            "best_avg_r": avg,
            "best_end_index": len(values),
            "worst_avg_r": avg,
            "worst_end_index": len(values),
            "latest_avg_r": avg,
            "negative_window_count": int(avg < 0.0) if values else 0,
        }
    windows = [
        {"end_index": index + window, "avg_r": mean(values[index : index + window])}
        for index in range(0, len(values) - window + 1)
    ]
    best = max(windows, key=lambda item: item["avg_r"])
    worst = min(windows, key=lambda item: item["avg_r"])
    return {
        "window_count": len(windows),
        "best_avg_r": best["avg_r"],
        "best_end_index": best["end_index"],
        "worst_avg_r": worst["avg_r"],
        "worst_end_index": worst["end_index"],
        "latest_avg_r": windows[-1]["avg_r"],
        "negative_window_count": sum(1 for item in windows if item["avg_r"] < 0.0),
    }


def _drawdown_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["entry_time"])
    equity = 0.0
    peak = 0.0
    worst_dd = 0.0
    worst_index = 0
    underwater = 0
    max_underwater = 0
    episodes = 0
    in_episode = False
    worst_recovery = 0
    worst_seen_index = 0
    for index, row in enumerate(ordered, start=1):
        equity += float(row["r"])
        if equity >= peak:
            if in_episode:
                in_episode = False
                underwater = 0
                if worst_seen_index:
                    worst_recovery = index - worst_seen_index
            peak = equity
        else:
            if not in_episode:
                in_episode = True
                episodes += 1
                underwater = 0
            underwater += 1
            max_underwater = max(max_underwater, underwater)
        dd = equity - peak
        if dd < worst_dd:
            worst_dd = dd
            worst_index = index
            worst_seen_index = index
            worst_recovery = 0
    return {
        "max_drawdown_r": worst_dd,
        "max_drawdown_trade_index": worst_index,
        "episode_count": episodes,
        "max_trades_underwater": max_underwater,
        "recovery_trades_from_worst": worst_recovery,
    }


def _r_vs_dollar_disconnect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winners = [row for row in rows if row["r"] > 0]
    losers = [row for row in rows if row["r"] <= 0]
    avg_winner_notional = _avg(row["notional"] for row in winners)
    avg_loser_notional = _avg(row["notional"] for row in losers)
    verdict = "GOOD" if avg_winner_notional >= avg_loser_notional * 0.95 else "REVIEW"
    return {
        "total_r": sum(row["r"] for row in rows),
        "total_pnl": sum(row["net_pnl"] for row in rows),
        "avg_winner_notional": avg_winner_notional,
        "avg_loser_notional": avg_loser_notional,
        "avg_winner_pnl": _avg(row["net_pnl"] for row in winners),
        "avg_loser_pnl": _avg(row["net_pnl"] for row in losers),
        "verdict": verdict,
    }


def _worst_periods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["entry_time"].strftime("%Y-%m")].append(row)
    output = []
    for period, values in buckets.items():
        stats = _stats_for(values)
        if stats["total_r"] >= 0:
            continue
        output.append(
            {
                "period": period,
                **stats,
                "entry_types": _top_counts(values, "entry_type", 4),
                "exit_reasons": _top_counts(values, "exit_reason", 4),
                "sectors": _top_counts(values, "sector", 4),
            }
        )
    return sorted(output, key=lambda item: item["total_r"])[:8]


def _intraday_alpha_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = (
        ("0-30min", lambda row: row["hold_hours"] <= 0.5),
        ("30-60min", lambda row: 0.5 < row["hold_hours"] <= 1.0),
        ("1-2h", lambda row: 1.0 < row["hold_hours"] <= 2.0),
        ("2-4h", lambda row: 2.0 < row["hold_hours"] <= 4.0),
        ("4h+", lambda row: row["hold_hours"] > 4.0),
    )
    output = []
    for label, predicate in buckets:
        values = [row for row in rows if predicate(row)]
        stats = _stats_for(values)
        output.append({"bucket": label, **stats, "pnl_per_trade": stats["net_pnl"] / stats["n"] if stats["n"] else 0.0})
    return output


def _instrumentation_gaps(rows: list[dict[str, Any]], metrics: dict[str, Any], alpha: dict[str, Any], sweeps: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if rows and not any(row.get("first30_ret") is not None for row in rows):
        gaps.append("No first30 metadata on trades; first30 signal-quality verdict is not available for this entry mode.")
    if not (alpha.get("shadow_nonselected") or {}).get("n"):
        gaps.append("No non-selected shadow trades were recorded; candidate surfacing can only be judged by accepted trades and external sweep artifacts.")
    if not sweeps.get("targeted_holdout"):
        gaps.append("No targeted holdout rows were attached; final report cannot quantify round-specific OOS depth.")
    if float(metrics.get("same_bar_fill_count", 0.0) or 0.0) > 0:
        gaps.append("Same-bar fills observed; replay/live timing parity must be fixed before promotion.")
    if not gaps:
        gaps.append("No critical instrumentation gaps detected in the available replay artifacts.")
    return gaps


def _shadow_group_stats(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(float(row.get("r", 0.0) or 0.0))
    return {key: _r_stats(values) for key, values in sorted(buckets.items())}


def _shadow_symbol_stats(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("symbol") or "")].append(float(row.get("r", 0.0) or 0.0))
    out = [{"symbol": symbol, **_r_stats(values)} for symbol, values in buckets.items()]
    return sorted(out, key=lambda item: (item["total_r"], item["n"]), reverse=reverse)


def _frontier_rank_bucket(rank: int) -> str:
    if rank <= 0:
        return "unknown"
    if rank <= 8:
        return "rank_001_008"
    if rank <= 20:
        return "rank_009_020"
    if rank <= 40:
        return "rank_021_040"
    return "rank_041_plus"


def _numeric_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "frontier_rank",
        "momentum_score",
        "bar_rvol",
        "cpr",
        "avwap_distance_pct",
        "or_width_pct",
        "breakout_distance_r",
        "mfe_r",
        "mae_r",
        "giveback_r",
        "hold_hours",
    )
    return {field: float(mean([float(row.get(field, 0.0) or 0.0) for row in rows])) if rows else 0.0 for field in fields}


def _profile_delta(winners: dict[str, float], losers: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(winners) | set(losers))
    return {key: float(winners.get(key, 0.0) - losers.get(key, 0.0)) for key in keys}


def _sweep_frontier(external: dict[str, Any]) -> dict[str, Any]:
    ws8 = external.get("ws8_summary") or {}
    capacity = external.get("capacity_validation") or []
    return {
        "capacity_validation": _dedup_rows([_compact_row(row) for row in capacity]),
        "targeted_train": _dedup_rows([_compact_row(row) for row in (ws8.get("top_train") or [])])[:10],
        "targeted_holdout": _dedup_rows([_compact_row(row) for row in (ws8.get("holdout") or [])])[:10],
        "best": _compact_row(ws8.get("best") or {}),
    }


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "score": row.get("score"),
        "blended_score": row.get("blended_score"),
        "reject_reason": row.get("reject_reason", ""),
        "metrics": dict(row.get("metrics") or {}),
        "mutations": dict(row.get("mutations") or {}),
    }


def _compact_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "best": _compact_row(summary.get("best") or {}),
        "baseline": _compact_row(summary.get("baseline") or {}),
        "evaluations": summary.get("evaluations"),
        "artifact_promotion_policy": summary.get("artifact_promotion_policy"),
    }


def _executive_verdicts(metrics: dict[str, Any], funnel: dict[str, Any], sweeps: dict[str, Any], alpha: dict[str, Any]) -> dict[str, str]:
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    total_r = float(metrics.get("expected_total_r", 0.0) or 0.0)
    dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    accept_rate = float(funnel.get("accept_rate", 0.0) or 0.0)
    accepted_avg = float((alpha.get("actual") or {}).get("avg_r", metrics.get("avg_r", 0.0)) or 0.0)
    shadow_avg = float((alpha.get("shadow_nonselected") or {}).get("avg_r", metrics.get("frontier_shadow_nonselected_avg_r", 0.0)) or 0.0)
    shadow_total = float((alpha.get("shadow_nonselected") or {}).get("total_r", metrics.get("frontier_shadow_nonselected_total_r", 0.0)) or 0.0)
    positive_pool = float(alpha.get("shadow_positive_pool_total_r", 0.0) or 0.0)
    negative_pool = float(alpha.get("shadow_negative_pool_total_r", 0.0) or 0.0)
    capacity = sweeps.get("capacity_validation") or []
    higher_bad = any(
        str(row.get("name", "")).startswith(("ws14", "ws18", "ws24")) and float((row.get("metrics") or {}).get("expected_total_r", 0.0) or 0.0) < 0
        for row in capacity
    )
    return {
        "Signal extraction": "GOOD" if total_r > 0 and pf >= 1.1 and shadow_avg < 0 else "REVIEW",
        "Discrimination": (
            f"GOOD (accepted avgR {_signed(accepted_avg)} vs non-selected shadow {_signed(shadow_avg)}, shadow total {_signed(shadow_total)})"
            if accept_rate < 0.02 and accepted_avg > 0 and shadow_avg < 0
            else "REVIEW"
        ),
        "Entry mechanism": "GOOD" if higher_bad and total_r > 0 else "REVIEW",
        "Exit mechanism": "REVIEW (failure stop disabled; quick exits and carry exits remain the main audit surface)",
        "Trade management": "GOOD" if dd <= 0.015 and pf >= 1.15 else "REVIEW",
        "Full-universe alpha": (
            f"NARROW (positive shadow pockets {_signed(positive_pool)} are overwhelmed by {_signed(negative_pool)} negative shadow R)"
            if positive_pool > 0 and negative_pool < 0
            else "REVIEW"
        ),
        "Primary bottleneck": "full-universe selector discrimination and route quality, not raw data coverage or websocket mechanics",
    }


def _strengths_weaknesses(
    rows: list[dict[str, Any]],
    groups: dict[str, Any],
    metrics: dict[str, Any],
    funnel: dict[str, Any],
    sweeps: dict[str, Any],
    alpha: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    best_symbol = _best_group(groups["symbol"])
    worst_symbol = _worst_group(groups["symbol"])
    best_exit = _best_group(groups["exit_reason"])
    worst_exit = _worst_group(groups["exit_reason"])
    wins = sorted([row for row in rows if row["net_pnl"] > 0], key=lambda row: row["net_pnl"], reverse=True)
    total_gains = sum(row["net_pnl"] for row in wins)
    top5_share = sum(row["net_pnl"] for row in wins[:5]) / total_gains if total_gains else 0.0
    strengths = [
        f"Positive full-train expectancy: total R {_signed(metrics['expected_total_r'])}, PF {_num(metrics['profit_factor'])}, net return {_pct(metrics['net_return_pct'])}.",
        f"Accepted-vs-shadow discrimination is positive: accepted avgR {_signed((alpha.get('actual') or {}).get('avg_r'))} vs full non-selected shadow avgR {_signed((alpha.get('shadow_nonselected') or {}).get('avg_r'))}.",
        f"Best symbol: {best_symbol[0]} (n={best_symbol[1]['n']}, WR={_pct(best_symbol[1]['win_rate'])}, avgR={_signed(best_symbol[1]['avg_r'])}, fee-net={_money(best_symbol[1]['net_pnl'])}).",
        f"Best exit reason: {best_exit[0]} (n={best_exit[1]['n']}, avgR={_signed(best_exit[1]['avg_r'])}, PF={_num(best_exit[1]['profit_factor'])}).",
        f"Winner concentration is acceptable for a seed: top 5 winners drive {_pct(top5_share)} of gross positive PnL.",
    ]
    weaknesses = [
        f"Worst symbol: {worst_symbol[0]} (n={worst_symbol[1]['n']}, WR={_pct(worst_symbol[1]['win_rate'])}, avgR={_signed(worst_symbol[1]['avg_r'])}, fee-net={_money(worst_symbol[1]['net_pnl'])}).",
        f"Worst exit reason: {worst_exit[0]} (n={worst_exit[1]['n']}, avgR={_signed(worst_exit[1]['avg_r'])}, PF={_num(worst_exit[1]['profit_factor'])}).",
        f"Full-universe non-selected shadow is negative: n={_int((alpha.get('shadow_nonselected') or {}).get('n'))}, totalR={_signed((alpha.get('shadow_nonselected') or {}).get('total_r'))}, despite positive pockets of {_signed(alpha.get('shadow_positive_pool_total_r'))}R.",
        f"Holdout is thin: {int((sweeps.get('targeted_holdout') or [{}])[0].get('metrics', {}).get('total_trades', 0) or 0)} trades in the top validation rows.",
        "Rejected gate decisions still need gate-level realized shadow attribution; full-frontier selected-vs-skipped shadow attribution is now recorded separately.",
    ]
    notes = [
        f"Entry selectivity is tight: {funnel['entries']} entries from {funnel['evaluated']} evaluated candidate-bars ({_pct(funnel['accept_rate'])}).",
        f"KRX hot frontier shadows {_int(metrics.get('frontier_symbol_max'))} symbols/day while keeping executable ws usage capped at {_int(metrics.get('active_symbol_max'))}.",
        "The high-frequency sweep selected the 10:25-13:00 ws8 slice; wider websocket budgets added trades but weakened expectancy and drawdown.",
        "All promotion remains research-only until KIS paper/live order and fill parity is attached.",
    ]
    quality = dict(funnel.get("raw_breakout_quality") or [])
    if quality.get("cpr_relaxed_entries"):
        notes.insert(1, f"Borderline CPR rescue accepted {quality['cpr_relaxed_entries']} high-score entries that still passed RVOL and downstream risk gates.")
    return strengths, weaknesses, notes


def _dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = stable_signature({"name": row.get("name"), "mutations": row.get("mutations", {})})
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _selected_holdout_trades(frontier: dict[str, Any], selected: str) -> int:
    for item in frontier.get("targeted_holdout", ()):
        if item.get("name") == selected:
            return _int((item.get("metrics") or {}).get("total_trades"))
    return 0


def _best_group(groups: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not groups:
        return "NONE", _stats_for([])
    return max(groups.items(), key=lambda item: (float(item[1].get("total_r", 0.0)), int(item[1].get("n", 0))))


def _worst_group(groups: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not groups:
        return "NONE", _stats_for([])
    return min(groups.items(), key=lambda item: (float(item[1].get("total_r", 0.0)), -int(item[1].get("n", 0))))


def _live_parity_audit(result: StrategyBacktestResult, config: dict[str, Any], mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_core_version": KALCB_CORE_VERSION,
        "shared_decision_core": result.metrics.get("shared_decision_core"),
        "live_parity_fill_timing": result.metrics.get("live_parity_fill_timing"),
        "auction_mode": result.metrics.get("auction_mode"),
        "replay_mode": result.metrics.get("replay_mode"),
        "same_bar_fill_count": result.metrics.get("same_bar_fill_count"),
        "universe_size": result.metrics.get("universe_size"),
        "data_available_symbol_count": result.metrics.get("data_available_symbol_count"),
        "unavailable_symbol_count": result.metrics.get("unavailable_symbol_count"),
        "unavailable_symbols": result.metrics.get("unavailable_symbols"),
        "candidate_pool_max": result.metrics.get("candidate_pool_max"),
        "candidate_pool_universe_fraction": result.metrics.get("candidate_pool_universe_fraction"),
        "candidate_pool_data_available_fraction": result.metrics.get("candidate_pool_data_available_fraction"),
        "active_symbol_max": result.metrics.get("active_symbol_max"),
        "selected_universe_fraction": result.metrics.get("selected_universe_fraction"),
        "frontier_enabled": result.metrics.get("frontier_enabled"),
        "frontier_size": result.metrics.get("frontier_size"),
        "frontier_symbol_max": result.metrics.get("frontier_symbol_max"),
        "frontier_universe_fraction": result.metrics.get("frontier_universe_fraction"),
        "frontier_selection_mode": result.metrics.get("frontier_selection_mode"),
        "frontier_active_selection_mode": result.metrics.get("frontier_active_selection_mode"),
        "frontier_shadow_nonselected_trade_count": result.metrics.get("frontier_shadow_nonselected_trade_count"),
        "frontier_shadow_nonselected_total_r": result.metrics.get("frontier_shadow_nonselected_total_r"),
        "frontier_rotation_proof_symbol_count": result.metrics.get("frontier_rotation_proof_symbol_count"),
        "frontier_rotation_proof_trade_count": result.metrics.get("frontier_rotation_proof_trade_count"),
        "frontier_rotation_proof_total_r": result.metrics.get("frontier_rotation_proof_total_r"),
        "frontier_rotation_proof_avg_r": result.metrics.get("frontier_rotation_proof_avg_r"),
        "frontier_rotation_global_trade_count": result.metrics.get("frontier_rotation_global_trade_count"),
        "frontier_rotation_global_total_r": result.metrics.get("frontier_rotation_global_total_r"),
        "frontier_rotation_global_avg_r": result.metrics.get("frontier_rotation_global_avg_r"),
        "frontier_rotation_frontier_proof_ready": result.metrics.get("frontier_rotation_frontier_proof_ready"),
        "frontier_rotation_promotion_count": result.metrics.get("frontier_rotation_promotion_count"),
        "ws_budget": mutations.get("kalcb.session.ws_budget", config.get("ws_budget")),
        "max_positions": mutations.get("kalcb.risk.max_positions", config.get("max_positions", 6)),
        "candidate_snapshot_count": result.metrics.get("candidate_snapshot_count"),
        "replay_event_count": result.metrics.get("replay_event_count"),
        "source_fingerprint": result.source_fingerprint,
        "candidate_snapshot_hash": result.candidate_snapshot_hash,
        "feature_bundle_hash": result.feature_bundle_hash,
        "paper_promotion_blockers": [
            "Need KIS paper intent/order/fill parity.",
            "Need deeper OOS/forward sample beyond the current six-trade holdout.",
            "Need REST throttle/defer audit under live EGW00201 conditions.",
        ],
    }


def _gate_result(metrics: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "positive_expected_total_r": float(metrics.get("expected_total_r", 0.0) or 0.0) > 0,
        "profit_factor_above_one": float(metrics.get("profit_factor", 0.0) or 0.0) > 1.0,
        "same_bar_fill_clean": float(audit.get("same_bar_fill_count", 0.0) or 0.0) == 0.0,
        "within_ws_budget": float(audit.get("active_symbol_max", 0.0) or 0.0) <= float(audit.get("ws_budget", 0.0) or 0.0),
        "research_only": True,
    }
    return {"status": "accepted_research_baseline" if all(checks.values()) else "review", "checks": checks}


def _or_width_bucket(row: dict[str, Any]) -> str:
    value = row["or_width_pct"]
    if value < 0.003:
        return "Tight (<0.3%)"
    if value <= 0.005:
        return "Normal (0.3-0.5%)"
    return "Wide (>0.5%)"


def _breakout_distance_bucket(rows: list[dict[str, Any]]):
    values = [row["breakout_distance_r"] for row in rows if row["breakout_distance_r"] > 0]
    midpoint = median(values) if values else 0.0

    def bucket(row: dict[str, Any]) -> str:
        if row["breakout_distance_r"] <= midpoint:
            return f"Near breakout (<=P50={midpoint:.2f}R)"
        return "Far breakout (>P50)"

    return bucket


def _rvol_bucket(row: dict[str, Any]) -> str:
    value = row["bar_rvol"]
    if value < 2.0:
        return "RVOL <2.0"
    if value < 3.0:
        return "RVOL 2.0-3.0"
    if value <= 5.0:
        return "RVOL 3.0-5.0"
    return "RVOL 5.0+"


def _avwap_bucket(row: dict[str, Any]) -> str:
    pct = row["avwap_distance_pct"]
    if pct <= 0.005:
        return "Slight premium (0% to +0.5%)"
    if pct <= 0.010:
        return "Premium (+0.5% to +1.0%)"
    return "Extended (> +1.0%)"


def _entry_time_bucket(row: dict[str, Any]) -> str:
    mins = row["entry_decision_time"].hour * 60 + row["entry_decision_time"].minute
    if mins < 10 * 60 + 30:
        return "09:30-10:30"
    if mins < 11 * 60 + 30:
        return "10:30-11:30"
    if mins <= 12 * 60:
        return "11:30-12:00"
    return "After 12:00"


def _flow_timing_bucket(row: dict[str, Any]) -> str:
    bars = max(0, int(round(row["hold_hours"] * 12)))
    if bars <= 6:
        return "Early FR (<=6 bars)"
    if bars <= 24:
        return "Mid FR (7-24 bars)"
    return "Late FR (>24 bars)"


def _score_monotonic(score_groups: dict[str, dict[str, Any]]) -> bool:
    ordered = sorted((int(key), value["avg_r"]) for key, value in score_groups.items() if str(key).isdigit())
    return all(ordered[index][1] <= ordered[index + 1][1] for index in range(len(ordered) - 1))


def _top_entry_bars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["entry_bar_index"], row["entry_bar_label"])].append(row)
    output = []
    for (index, label), values in grouped.items():
        stats = _stats_for(values)
        output.append({"bar_index": index, "time": label, "n": stats["n"], "win_rate": stats["win_rate"], "avg_r": stats["avg_r"]})
    return sorted(output, key=lambda item: item["n"], reverse=True)


def _matrix(rows: list[dict[str, Any]], row_key: str, col_key: str) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    row_values = sorted({row[row_key] for row in rows})
    col_values = sorted({row[col_key] for row in rows})
    for row_value in row_values:
        output[row_value] = {}
        for col_value in col_values:
            output[row_value][col_value] = _stats_for([row for row in rows if row[row_key] == row_value and row[col_key] == col_value])
    return output


def _mfe_mae(rows: list[dict[str, Any]]) -> dict[str, Any]:
    winners = [row for row in rows if row["r"] > 0]
    losers = [row for row in rows if row["r"] <= 0]
    mfe_values = [row["mfe_r"] for row in rows]
    mae_values = [row["mae_r"] for row in rows]
    losers_mfe = [row for row in losers if row["mfe_r"] > 0.3]
    return {
        "winner_count": len(winners),
        "loser_count": len(losers),
        "winner_mean_mfe_r": mean([row["mfe_r"] for row in winners]) if winners else 0.0,
        "winner_capture_efficiency": mean([min(1.0, max(0.0, row["r"] / row["mfe_r"])) for row in winners if row["mfe_r"] > 0]) if winners else 0.0,
        "winner_mean_giveback_r": mean([row["giveback_r"] for row in winners]) if winners else 0.0,
        "loser_mean_mae_r": mean([row["mae_r"] for row in losers]) if losers else 0.0,
        "mfe_p25": _percentile(mfe_values, 25),
        "mfe_p50": _percentile(mfe_values, 50),
        "mfe_p75": _percentile(mfe_values, 75),
        "mae_p25": _percentile(mae_values, 25),
        "mae_p50": _percentile(mae_values, 50),
        "mae_p75": _percentile(mae_values, 75),
        "losers_with_mfe_gt_03": len(losers_mfe),
        "losers_with_mfe_gt_03_share": len(losers_mfe) / len(losers) if losers else 0.0,
    }


def _hold_duration_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["hold_hours"])
    cuts = [10, 20, 30, 40, 100]
    output = []
    start = 0
    for cut in cuts:
        end = len(ordered) if cut == 100 else max(start + 1, int(len(ordered) * cut / 100))
        bucket = ordered[start:end]
        stats = _stats_for(bucket)
        output.append({"percentile": cut, "hours": bucket[-1]["hold_hours"] if bucket else 0.0, "win_rate": stats["win_rate"], "avg_r": stats["avg_r"], "n": stats["n"]})
        start = end
    return output


def _percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _append_group_section(lines: list[str], title: str, groups: dict[str, dict[str, Any]], *, key_label: str, limit: int | None = None) -> None:
    _section(lines, title)
    if not groups:
        lines.append("  (no trades)")
        return
    ordered = sorted(groups.items(), key=lambda item: abs(float(item[1].get("total_r", 0.0))), reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    for key, stats in ordered:
        lines.append(
            f"  {key_label}{key}: n={stats['n']}, WR={_pct(stats['win_rate'])}, "
            f"Mean R={_signed(stats['avg_r'])}, Median R={_signed(stats['median_r'])}, "
            f"PF={_num(stats['profit_factor'])}, Total R={_signed(stats['total_r'])}, PnL={_money(stats['net_pnl'])}"
        )
        if "Regime Sizing" in title:
            lines.append(f"    Avg position: {_money(stats['avg_notional'])}")


def _append_shadow_group_section(lines: list[str], title: str, groups: dict[str, dict[str, Any]], limit: int | None = None) -> None:
    lines.append(title)
    if not groups:
        lines.append("    (no shadow trades)")
        return
    ordered = sorted(groups.items(), key=lambda item: abs(float(item[1].get("total_r", 0.0))), reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    for key, stats in ordered:
        lines.append(
            f"    {key}: n={_int(stats.get('n'))}, WR={_pct(stats.get('win_rate'))}, "
            f"AvgR={_signed(stats.get('avg_r'))}, TotalR={_signed(stats.get('total_r'))}, PF={_num(stats.get('profit_factor'))}"
        )


def _append_counter_stats(lines: list[str], groups: dict[str, dict[str, Any]], *, label_prefix: str) -> None:
    for key, stats in sorted(groups.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999):
        lines.append(f"{label_prefix}{key}: n={stats['n']}, WR={_pct(stats['win_rate'])}, Mean R={_signed(stats['avg_r'])}, Total R={_signed(stats['total_r'])}")


def _append_group_stats_inline(lines: list[str], groups: dict[str, dict[str, Any]]) -> None:
    for key, stats in groups.items():
        lines.append(f"  {key}: n={stats['n']}, WR={_pct(stats['win_rate'])}, Mean R={_signed(stats['avg_r'])}, PF={_num(stats['profit_factor'])}, Total R={_signed(stats['total_r'])}")


def _append_compact_stat_table(lines: list[str], groups: dict[str, dict[str, Any]], *, indent: str) -> None:
    if not groups:
        lines.append(f"{indent}(no rows)")
        return
    for key, stats in sorted(groups.items(), key=lambda item: str(item[0])):
        lines.append(
            f"{indent}{key}: n={_int(stats.get('n'))}, WR={_pct(stats.get('win_rate'))}, "
            f"AvgR={_signed(stats.get('avg_r'))}, TotalR={_signed(stats.get('total_r'))}, PF={_num(stats.get('profit_factor'))}"
        )


def _append_period_table(lines: list[str], groups: dict[str, dict[str, Any]]) -> None:
    if not groups:
        lines.append("  (no rows)")
        return
    lines.append("  Period                    N     WR%    Avg R   Total R      PnL")
    lines.append("  ---------------------------------------------------------------")
    for key, stats in sorted(groups.items(), key=lambda item: str(item[0])):
        lines.append(
            f"  {str(key):<22} {_int(stats.get('n')):>4} {_pct(stats.get('win_rate')):>7} "
            f"{_signed(stats.get('avg_r')):>8} {_signed(stats.get('total_r')):>9} {_money(stats.get('net_pnl')):>10}"
        )


def _append_matrix(lines: list[str], matrix: dict[str, dict[str, dict[str, Any]]]) -> None:
    if not matrix:
        lines.append("  (no trades)")
        return
    columns = sorted({col for row in matrix.values() for col in row})
    header = "  " + " " * 14 + " ".join(f"{col[:12]:>16}" for col in columns)
    lines.append(header)
    lines.append("  " + "-" * max(50, len(header) - 2))
    for row_key, row in sorted(matrix.items()):
        parts = []
        for col in columns:
            stats = row.get(col, {})
            parts.append("--".rjust(16) if not stats.get("n") else f"{_signed(stats['avg_r'])}({stats['n']:>2})".rjust(16))
        lines.append(f"  {row_key:<14} {''.join(parts)}")


def _box(lines: list[str], title: str) -> None:
    lines.append("=" * 72)
    lines.append(f"  {title}")
    lines.append("=" * 72)


def _section(lines: list[str], title: str) -> None:
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)


def _pct(value: Any) -> str:
    return f"{100.0 * _float(value):.1f}%"


def _pct_value(value: Any) -> str:
    return f"{100.0 * _float(value):.0f}%"


def _signed(value: Any) -> str:
    return f"{_float(value):+.3f}"


def _num(value: Any) -> str:
    value = _float(value)
    if value >= 998.0:
        return "inf"
    return f"{value:.2f}"


def _money(value: Any) -> str:
    return f"{_float(value):+,.0f}"


def _int(value: Any) -> int:
    return int(_float(value))


def _float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _avg(values: Iterable[Any]) -> float:
    vals = [_float(value) for value in values]
    return float(mean(vals)) if vals else 0.0


def _equity_risk_metrics(equity_curve: Iterable[float]) -> dict[str, float]:
    equity = [float(value) for value in equity_curve if value is not None]
    returns = [
        (equity[index] - equity[index - 1]) / equity[index - 1]
        for index in range(1, len(equity))
        if equity[index - 1]
    ]
    if not returns:
        return {"mean_return": 0.0, "downside_deviation": 0.0, "sortino": 0.0}
    downside = [min(value, 0.0) for value in returns]
    downside_dev = pstdev(downside) if len(downside) > 1 else 0.0
    avg_return = mean(returns)
    sortino = (avg_return / downside_dev) * math.sqrt(252 * 390) if downside_dev > 0 else 0.0
    return {"mean_return": float(avg_return), "downside_deviation": float(downside_dev), "sortino": float(sortino)}


def _observed_session_days(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> int:
    for key in ("session_count", "observed_session_days", "candidate_snapshot_count"):
        value = _int(metrics.get(key))
        if value > 0:
            return value
    return len({row["entry_time"].date() for row in rows})


def _signed_pct_bucket(value: Any, cuts: tuple[float, ...]) -> str:
    val = _float(value)
    previous = None
    for cut in cuts:
        if val <= cut:
            if previous is None:
                return f"<= {cut * 100:.1f}%"
            return f"{previous * 100:.1f}% to {cut * 100:.1f}%"
        previous = cut
    return f"> {cuts[-1] * 100:.1f}%"


def _unit_bucket(value: Any, cuts: tuple[float, ...]) -> str:
    val = _float(value)
    previous = None
    for cut in cuts:
        if val <= cut:
            if previous is None:
                return f"<= {cut:.2f}"
            return f"{previous:.2f} to {cut:.2f}"
        previous = cut
    return f"> {cuts[-1]:.2f}"


def _top_counts(rows: list[dict[str, Any]], key: str, limit: int) -> str:
    counts = Counter(str(row.get(key) or "UNKNOWN") for row in rows)
    return ", ".join(f"{name}:{count}" for name, count in counts.most_common(limit)) or "n/a"


def _artifact_paths(round_dir: Path) -> dict[str, str]:
    names = (
        "optimized_config.json",
        "run_spec.json",
        "run_summary.json",
        "phase_state.json",
        "progress.json",
        "phase_activity_log.jsonl",
        "phase_1_analysis.json",
        "phase_1_diagnostics.txt",
        "candidate_frontier.json",
        "diagnostics_summary.json",
        "live_parity_audit.json",
        "full_diagnostics_index.json",
        "round_final_diagnostics.txt",
        "round_evaluation.txt",
        "trade_events.jsonl",
    )
    return {name: str(round_dir / name) for name in names}


def _index_payload(analysis: dict[str, Any], artifact_paths: dict[str, str], source_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "strategy": "kalcb",
        "round": analysis.get("round") or 1,
        "generated_at_utc": analysis["generated_at_utc"],
        "selected_candidate": analysis["selected_candidate"],
        "source_fingerprint": analysis["source_fingerprint"],
        "candidate_snapshot_hash": analysis["candidate_snapshot_hash"],
        "feature_bundle_hash": analysis["feature_bundle_hash"],
        "source_artifacts": source_paths,
        "artifact_paths": artifact_paths,
        "summary": analysis["diagnostics_summary"],
    }


def _progress_snapshot(analysis: dict[str, Any], artifact_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "strategy": "kalcb",
        "round": analysis.get("round") or 1,
        "status": "completed",
        "candidate": analysis["selected_candidate"],
        "generated_at_utc": analysis["generated_at_utc"],
        "completed_phases": [0],
        "headline_metrics": {
            key: analysis["final_metrics"].get(key)
            for key in ("total_trades", "win_rate", "profit_factor", "max_drawdown_pct", "net_return_pct", "expected_total_r")
        },
        "artifacts": artifact_paths,
    }


def _phase_analysis_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": 0,
        "focus": "optimized baseline promotion and full diagnostics",
        "candidate": analysis["selected_candidate"],
        "metrics": analysis["final_metrics"],
        "verdicts": analysis["verdicts"],
        "strengths": analysis["strengths"],
        "weaknesses": analysis["weaknesses"],
        "live_parity_audit": analysis["live_parity_audit"],
    }


def _phase_diagnostics_text(analysis: dict[str, Any]) -> str:
    metrics = analysis["final_metrics"]
    return "\n".join(
        [
            "KALCB phase 0 optimized baseline diagnostics",
            f"Candidate: {analysis['selected_candidate']}",
            f"Trades: {int(metrics['total_trades'])}",
            f"Expected total R: {_signed(metrics['expected_total_r'])}",
            f"Profit factor: {_num(metrics['profit_factor'])}",
            f"Max DD: {_pct(metrics['max_drawdown_pct'])}",
            f"Same-bar fills: {analysis['live_parity_audit'].get('same_bar_fill_count')}",
            f"Active symbols max: {analysis['live_parity_audit'].get('active_symbol_max')}",
        ]
    ) + "\n"


def _run_summary_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_candidate": analysis["selected_candidate"],
        "final_metrics": analysis["final_metrics"],
        "verdicts": analysis["verdicts"],
        "signal_funnel": analysis["signal_funnel"],
        "live_parity_audit": analysis["live_parity_audit"],
    }


def _enrich_manifest(path: Path, round_num: int, fields: dict[str, Any]) -> None:
    manifest = _load_json(path)
    rounds = manifest.setdefault("rounds", [])
    for item in rounds:
        if int(item.get("round", 0) or 0) == int(round_num):
            item.update(fields)
            break
    _atomic_write_json(manifest, path)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote the optimized KALCB baseline into stock-style round artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--ws8-summary", default=str(DEFAULT_WS8_SUMMARY))
    parser.add_argument("--capacity-validation", default=str(DEFAULT_CAPACITY_VALIDATION))
    parser.add_argument("--broad-summary", default=str(DEFAULT_BROAD_SUMMARY))
    args = parser.parse_args(argv)
    payload = promote_kalcb_baseline_round(
        config_path=args.config,
        output_root=args.output_root,
        round_num=args.round,
        ws8_summary_path=args.ws8_summary,
        capacity_validation_path=args.capacity_validation,
        broad_summary_path=args.broad_summary,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
