from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.auto.oos_ablation import (
    KALCBFixedTradePlanAblationAdapter,
    RoundArtifact,
    clean_metric_row,
    resolve_windows,
)
from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_runner import _plugin_artifact_metadata
from backtests.auto.shared.phase_state import PhaseState, _atomic_write_json
from backtests.auto.shared.round_manager import RoundManager
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.kalcb.fixed_trade_plan_phase import KALCBFixedTradePlanOptimizationPlugin

DEFAULT_CONFIG = ROOT / "config/optimization/kalcb.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "data/backtests/output"
DEFAULT_CANDIDATE = (
    ROOT
    / "data/backtests/output/kalcb/round_2/oos_high_notional_target_20260527/recommended_mutations.json"
)
ROUND_NUM = 3
ROUND_NAME = "round_3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote hn_notional1p0_target55p0 into KALCB round_3 artifacts."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--keep-missing-later-manifest-rounds",
        action="store_true",
        help="Preserve manifest entries after round_3 even if their round directories are missing.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_root = Path(args.output_root)
    candidate_path = Path(args.candidate)
    manager = RoundManager("stock", "kalcb", base_dir=output_root)
    round_dir = manager.get_round_dir(ROUND_NUM)
    round_dir.mkdir(parents=True, exist_ok=True)

    config = normalize_runtime_config("kalcb", load_yaml_config(config_path))
    config["fixed_trade_plan_phase_auto"] = True
    config["force_rebuild_cache"] = False
    config["skip_initial_baseline_eval"] = False
    config.setdefault("promotion_status", "research_only")

    candidate = _read_json(candidate_path)
    label = str(candidate.get("label") or "hn_notional1p0_target55p0")
    train_payload = dict(candidate.get("train") or {})
    oos_payload = dict(candidate.get("oos") or {})
    mutations = dict(train_payload.get("mutations") or {})
    if not mutations:
        raise ValueError(f"No train.mutations found in {candidate_path}")
    mutations.update(
        {
            "kalcb.exit.quick_exit_enabled": True,
            "kalcb.exit.quick_exit_bars": 10,
            "kalcb.exit.quick_exit_min_r": -0.5,
            "kalcb.exit.stop_pct": 0.0045,
            "kalcb.exit.target_r": 55.0,
            "kalcb.risk.max_position_notional_pct": 1.0,
        }
    )
    config["fixed_candidate_source"] = {
        "path": mutations["_kalcb.source.path"],
        "section": mutations.get("_kalcb.source.section", "top_portfolio_proxy"),
        "rank": mutations.get("_kalcb.source.rank", 0),
    }

    round2 = _load_round_artifact(manager, 2)
    round2_mutations = dict(round2.optimized.get("mutations") or round2.run_summary.get("cumulative_mutations") or {})
    mutation_delta = _mutation_delta(round2_mutations, mutations)

    target_artifact = RoundArtifact(
        round_num=ROUND_NUM,
        path=round_dir,
        optimized={
            "strategy": "kalcb",
            "round": ROUND_NUM,
            "round_name": ROUND_NAME,
            "mutations": mutations,
        },
        run_summary={"strategy": "kalcb", "round": ROUND_NUM, "round_name": ROUND_NAME},
    )
    windows = resolve_windows(config, target_artifact)

    plugin = KALCBFixedTradePlanOptimizationPlugin(
        config,
        output_dir=round_dir,
        max_workers=max(1, int(args.max_workers)),
        capability_level=str(config.get("capability_level") or "real_replay"),
    )
    final_metrics = plugin.compute_final_metrics(mutations)
    _add_canonical_metric_aliases(final_metrics)

    state = PhaseState(
        current_phase=1,
        completed_phases=[1],
        cumulative_mutations=mutations,
        phase_results={
            1: {
                "phase": 1,
                "focus": "round_3 OOS-ablation promotion of quick-exit, 45bp fixed stop, 55R target, and 100% max notional cap",
                "applied_phase_mutations": True,
                "accepted_count": 1,
                "kept_features": [
                    "combo_stop0045_quick_exit",
                    "risk.max_position_notional_pct=1.0",
                    "exit.target_r=55.0",
                ],
                "final_metrics": final_metrics,
                "holdout_metrics": dict(oos_payload.get("metrics") or {}),
                "rounds": [
                    {
                        "round_num": 1,
                        "best_name": label,
                        "best_score": final_metrics.get("immutable_score"),
                        "best_delta_pct": _pct_delta(
                            final_metrics.get("broker_net_return_pct"),
                            (round2.run_summary.get("final_metrics") or {}).get("broker_net_return_pct"),
                        ),
                        "kept": True,
                        "rejected_count": 0,
                    }
                ],
                "source_artifacts": _source_artifacts(candidate_path),
            }
        },
        phase_gate_results={
            1: {
                "passed": _hygiene_pass(final_metrics),
                "candidate": label,
                "same_bar_fill_count": final_metrics.get("same_bar_fill_count"),
                "end_open_position_count": final_metrics.get("end_open_position_count"),
                "broker_max_drawdown_pct": final_metrics.get("broker_max_drawdown_pct"),
            }
        },
        round_name=ROUND_NAME,
    )
    diagnostics_summary = plugin.write_full_diagnostics(
        state,
        round_dir,
        round_num=ROUND_NUM,
        round_name=ROUND_NAME,
    )

    adapter = KALCBFixedTradePlanAblationAdapter(config, windows, target_artifact, round_dir / "holdout_replay")
    oos_evaluation = adapter.evaluate(label, mutations, "oos")
    train_confirmation = {
        "label": label,
        "window": "train",
        "metrics": final_metrics,
        "metric_row": clean_metric_row(final_metrics),
        "source": dict(train_payload.get("source") or {}),
    }
    _atomic_write_json(
        {
            "strategy": "kalcb",
            "round": ROUND_NUM,
            "round_name": ROUND_NAME,
            "selected_candidate": label,
            "generated_at_utc": _utc_now(),
            "windows": {
                "train": {"start": windows.train_start, "end": windows.train_end},
                "holdout": {"start": windows.oos_start, "end": windows.oos_end},
            },
            "train": train_confirmation,
            "holdout": oos_evaluation,
            "source_candidate_payload": str(candidate_path),
            "source_artifacts": _source_artifacts(candidate_path),
        },
        round_dir / "holdout_evaluation.json",
    )

    live_parity_audit = _read_json(round_dir / "live_parity_audit.json")
    artifact_metadata = _round_artifact_metadata(
        plugin=plugin,
        state=state,
        final_metrics=final_metrics,
        diagnostics_summary=diagnostics_summary,
        live_parity_audit=live_parity_audit,
        oos_evaluation=oos_evaluation,
        candidate_path=candidate_path,
        label=label,
        mutation_delta=mutation_delta,
        windows=windows,
    )
    manager.write_run_spec(
        round_dir,
        ROUND_NUM,
        strategy_name="kalcb",
        description=(
            "Round 3 promotion of hn_notional1p0_target55p0 from the round_2 OOS ablation "
            "and high-notional target refinement campaign."
        ),
        baseline_mutations=round2_mutations,
        baseline_source=manager.optimized_config_path(manager.round_path(2)),
        baseline_metadata={
            "round_2_run_summary": str(manager.run_summary_path(manager.round_path(2))),
            "round_2_final_metrics": round2.run_summary.get("final_metrics") or {},
            "mutation_delta_from_round_2": mutation_delta,
        },
        execution_context={
            "config_path": str(config_path),
            "selected_candidate": label,
            "candidate_payload": str(candidate_path),
            "windows": artifact_metadata["windows"],
            "live_backtest_parity_alignment": artifact_metadata["live_backtest_parity_alignment"],
        },
        overwrite=True,
    )
    manager.write_run_summary(
        round_dir,
        mutations,
        final_metrics,
        state.completed_phases,
        round_num=ROUND_NUM,
        artifact_metadata=artifact_metadata,
    )
    manager.write_optimized_config(round_dir, mutations, artifact_metadata=artifact_metadata)
    manager.append_to_manifest(ROUND_NUM, mutations, final_metrics, artifact_metadata=artifact_metadata)
    _enrich_manifest(
        manager,
        round_num=ROUND_NUM,
        artifact_metadata=artifact_metadata,
        final_metrics=final_metrics,
        oos_metrics=dict(oos_evaluation.get("metrics") or {}),
        prune_missing_later=not args.keep_missing_later_manifest_rounds,
    )
    _write_round_evaluation(
        round_dir / "round_evaluation.txt",
        label=label,
        train=final_metrics,
        holdout=dict(oos_evaluation.get("metrics") or {}),
        mutation_delta=mutation_delta,
        live_parity_audit=live_parity_audit,
    )
    _append_round_final_holdout_addendum(
        round_dir / "round_final_diagnostics.txt",
        label=label,
        holdout=dict(oos_evaluation.get("metrics") or {}),
        live_parity_audit=live_parity_audit,
    )
    _atomic_write_json(
        {
            "strategy": "kalcb",
            "round": ROUND_NUM,
            "round_name": ROUND_NAME,
            "selected_candidate": label,
            "generated_at_utc": _utc_now(),
            "optimized_config": str(manager.optimized_config_path(round_dir)),
            "run_summary": str(manager.run_summary_path(round_dir)),
            "round_final_diagnostics": str(round_dir / "round_final_diagnostics.txt"),
            "diagnostics_summary": str(round_dir / "diagnostics_summary.json"),
            "live_parity_audit": str(round_dir / "live_parity_audit.json"),
            "paper_live_parity_contract": str(round_dir / "paper_live_parity_contract.json"),
            "holdout_evaluation": str(round_dir / "holdout_evaluation.json"),
            "manifest": str(manager.manifest_path),
            "train_headline": _headline_metrics(final_metrics),
            "holdout_headline": _headline_metrics(dict(oos_evaluation.get("metrics") or {})),
            "mutation_delta_from_round_2": mutation_delta,
        },
        round_dir / "promotion_summary.json",
    )
    print(
        json.dumps(
            {
                "strategy": "kalcb",
                "round": ROUND_NUM,
                "selected_candidate": label,
                "round_dir": str(round_dir),
                "manifest": str(manager.manifest_path),
                "train": _headline_metrics(final_metrics),
                "holdout": _headline_metrics(dict(oos_evaluation.get("metrics") or {})),
                "live_parity_status": live_parity_audit.get("status"),
                "manifest_latest_active_round": _manifest_latest_round(manager.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_round_artifact(manager: RoundManager, round_num: int) -> Any:
    round_dir = manager.round_path(round_num)
    return RoundArtifact(
        round_num=round_num,
        path=round_dir,
        optimized=_read_json(manager.optimized_config_path(round_dir)),
        run_summary=_read_json(manager.run_summary_path(round_dir)),
        diagnostics=_read_json(round_dir / "diagnostics_summary.json"),
        phase_state=_read_json(round_dir / "phase_state.json"),
    )


def _round_artifact_metadata(
    *,
    plugin: KALCBFixedTradePlanOptimizationPlugin,
    state: PhaseState,
    final_metrics: dict[str, Any],
    diagnostics_summary: dict[str, Any],
    live_parity_audit: dict[str, Any],
    oos_evaluation: dict[str, Any],
    candidate_path: Path,
    label: str,
    mutation_delta: dict[str, Any],
    windows: Any,
) -> dict[str, Any]:
    metadata = _plugin_artifact_metadata(plugin, state, final_metrics)
    oos_metrics = dict(oos_evaluation.get("metrics") or {})
    metadata.update(
        {
            "family": "stock",
            "strategy": "kalcb",
            "round": ROUND_NUM,
            "round_name": ROUND_NAME,
            "selected_candidate": label,
            "candidate_kind": "high_notional_target_refine",
            "source_candidate_payload": str(candidate_path),
            "source_ablation_artifacts": _source_artifacts(candidate_path),
            "mutation_delta_from_round_2": mutation_delta,
            "mutation_hash": stable_signature(state.cumulative_mutations),
            "selection_metrics": _headline_metrics(final_metrics),
            "oos_validation": {
                "label": label,
                "window": "holdout",
                "start": windows.oos_start,
                "end": windows.oos_end,
                "metrics": _headline_metrics(oos_metrics),
                "audit_pass": _hygiene_pass(oos_metrics),
                "source": dict(oos_evaluation.get("source") or {}),
            },
            "holdout_window_start": windows.oos_start,
            "holdout_window_end": windows.oos_end,
            "holdout_net_return_pct": oos_metrics.get("broker_net_return_pct"),
            "holdout_official_mtm_net_return_pct": oos_metrics.get("official_mtm_net_return_pct"),
            "holdout_trade_count": oos_metrics.get("trade_count", oos_metrics.get("total_trades")),
            "holdout_win_rate": oos_metrics.get("win_rate"),
            "holdout_max_drawdown_pct": oos_metrics.get("broker_max_drawdown_pct"),
            "holdout_avg_mfe_capture": oos_metrics.get("avg_mfe_capture"),
            "holdout_audit_pass": _hygiene_pass(oos_metrics),
            "diagnostics": {
                "round_final_diagnostics": str(ROOT / "data/backtests/output/kalcb/round_3/round_final_diagnostics.txt"),
                "diagnostics_summary": str(ROOT / "data/backtests/output/kalcb/round_3/diagnostics_summary.json"),
                "live_parity_audit": str(ROOT / "data/backtests/output/kalcb/round_3/live_parity_audit.json"),
                "paper_live_parity_contract": str(ROOT / "data/backtests/output/kalcb/round_3/paper_live_parity_contract.json"),
                "holdout_evaluation": str(ROOT / "data/backtests/output/kalcb/round_3/holdout_evaluation.json"),
            },
            "diagnostics_summary": {
                "final": diagnostics_summary.get("final"),
                "delta_vs_baseline": diagnostics_summary.get("delta_vs_baseline"),
                "holdout_excluded": diagnostics_summary.get("holdout_excluded"),
            },
            "windows": {
                "train": {"start": windows.train_start, "end": windows.train_end},
                "holdout": {"start": windows.oos_start, "end": windows.oos_end},
            },
            "live_backtest_parity_alignment": _live_backtest_parity_alignment(live_parity_audit),
        }
    )
    return metadata


def _source_artifacts(candidate_path: Path) -> dict[str, str]:
    base = candidate_path.parent
    return {
        "recommended_mutations": str(candidate_path),
        "oos_ablation_results": str(base / "oos_ablation_results.json"),
        "oos_ablation_summary": str(base / "oos_ablation_summary.md"),
    }


def _enrich_manifest(
    manager: RoundManager,
    *,
    round_num: int,
    artifact_metadata: dict[str, Any],
    final_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    prune_missing_later: bool,
) -> None:
    manifest_path = manager.manifest_path
    if manifest_path.exists():
        backup = manager.strategy_dir / f"rounds_manifest.pre_round3_promotion.{_stamp()}.json"
        shutil.copy2(manifest_path, backup)
    manifest = manager.load_manifest()
    manifest["family"] = "stock"
    manifest["strategy"] = "kalcb"
    manifest["generated_at_utc"] = _utc_now()
    rounds = []
    removed_missing_later: list[int] = []
    for item in manifest.get("rounds", []):
        item_round = int(item.get("round", 0) or 0)
        if prune_missing_later and item_round > round_num and not manager.round_path(item_round).exists():
            removed_missing_later.append(item_round)
            continue
        rounds.append(item)
    manifest["rounds"] = rounds
    entry = next(item for item in manifest["rounds"] if int(item.get("round", 0) or 0) == round_num)
    entry.update(
        {
            "strategy": "kalcb",
            "round_name": ROUND_NAME,
            "selected_candidate": artifact_metadata["selected_candidate"],
            "candidate_kind": artifact_metadata["candidate_kind"],
            "mutation_hash": artifact_metadata["mutation_hash"],
            "mutation_delta_from_round_2": artifact_metadata["mutation_delta_from_round_2"],
            "source_ablation_artifacts": artifact_metadata["source_ablation_artifacts"],
            "selection_metrics": artifact_metadata["selection_metrics"],
            "holdout_window_start": artifact_metadata["holdout_window_start"],
            "holdout_window_end": artifact_metadata["holdout_window_end"],
            "holdout_net_return_pct": oos_metrics.get("broker_net_return_pct"),
            "holdout_official_mtm_net_return_pct": oos_metrics.get("official_mtm_net_return_pct"),
            "holdout_trade_count": oos_metrics.get("trade_count", oos_metrics.get("total_trades")),
            "holdout_win_rate": oos_metrics.get("win_rate"),
            "holdout_max_drawdown_pct": oos_metrics.get("broker_max_drawdown_pct"),
            "holdout_avg_mfe_capture": oos_metrics.get("avg_mfe_capture"),
            "holdout_audit_pass": artifact_metadata["holdout_audit_pass"],
            "oos_validation": artifact_metadata["oos_validation"],
            "live_backtest_parity_alignment": artifact_metadata["live_backtest_parity_alignment"],
            "diagnostics": artifact_metadata["diagnostics"],
            "manifest_cleanup": {
                "removed_missing_later_round_entries": removed_missing_later,
                "reason": "round_3 is now the latest active optimized KALCB round; later manifest entries had no round directories.",
            },
            "updated_at_utc": _utc_now(),
            "net_return_pct": final_metrics.get("broker_net_return_pct"),
            "max_drawdown_pct": final_metrics.get("broker_max_drawdown_pct"),
        }
    )
    manifest["rounds"].sort(key=lambda item: int(item.get("round", 0) or 0))
    _atomic_write_json(manifest, manifest_path)


def _write_round_evaluation(
    path: Path,
    *,
    label: str,
    train: dict[str, Any],
    holdout: dict[str, Any],
    mutation_delta: dict[str, Any],
    live_parity_audit: dict[str, Any],
) -> None:
    lines = [
        "KALCB round_3 promotion evaluation",
        "",
        f"Selected candidate: {label}",
        "",
        "Train shared-core replay:",
        f"  net={_fmt_pct(train.get('broker_net_return_pct'))}, MTM={_fmt_pct(train.get('official_mtm_net_return_pct'))}, DD={_fmt_pct(train.get('broker_max_drawdown_pct'))}, trades={_fmt_num(train.get('trade_count'))}, WR={_fmt_pct(train.get('win_rate'))}",
        "",
        "Locked holdout replay:",
        f"  net={_fmt_pct(holdout.get('broker_net_return_pct'))}, MTM={_fmt_pct(holdout.get('official_mtm_net_return_pct'))}, DD={_fmt_pct(holdout.get('broker_max_drawdown_pct'))}, trades={_fmt_num(holdout.get('trade_count'))}, WR={_fmt_pct(holdout.get('win_rate'))}",
        "",
        "Mutation delta from round_2:",
    ]
    for key, value in mutation_delta.items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "Live/backtest parity alignment:",
            f"  status={live_parity_audit.get('status')}",
            f"  same_bar_fill_count={live_parity_audit.get('backtest_hash_baselines', {}).get('same_bar_fill_count')}",
            f"  fill_timing={live_parity_audit.get('expected_contract', {}).get('fill_timing')}",
            f"  auction_mode={live_parity_audit.get('expected_contract', {}).get('auction_mode')}",
            "",
            "Deployment note: artifact remains research-only until paper/live hash evidence matches the contract.",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _append_round_final_holdout_addendum(
    path: Path,
    *,
    label: str,
    holdout: dict[str, Any],
    live_parity_audit: dict[str, Any],
) -> None:
    marker = "## Locked Holdout Addendum"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    lines = [
        marker,
        "",
        f"Selected candidate: {label}",
        f"Holdout official MTM: {_fmt_pct(holdout.get('official_mtm_net_return_pct'))}",
        f"Holdout closed-trade net: {_fmt_pct(holdout.get('broker_net_return_pct'))}",
        f"Holdout max drawdown: {_fmt_pct(holdout.get('broker_max_drawdown_pct'))}",
        f"Holdout trades: {_fmt_num(holdout.get('trade_count'))}",
        f"Holdout win rate: {_fmt_pct(holdout.get('win_rate'))}",
        f"Holdout audit pass: {_hygiene_pass(holdout)}",
        f"Same-bar fills: {_fmt_num(holdout.get('same_bar_fill_count'))}",
        f"Forced replay closes: {_fmt_num(holdout.get('forced_replay_close_count'))}",
        f"Rejected orders: {_fmt_num(holdout.get('rejected_order_count'))}",
        f"End open positions: {_fmt_num(holdout.get('end_open_position_count'))}",
        f"Paper/live parity status: {live_parity_audit.get('status')}",
        "",
        "Production promotion remains blocked until paper/live decision, action, fill, trade, state, snapshot, and bar hashes match the contract.",
        "",
    ]
    prefix = existing.rstrip()
    text = ("\n".join((prefix, "", *lines)) if prefix else "\n".join(lines))
    _atomic_write_text(path, text)


def _live_backtest_parity_alignment(audit: dict[str, Any]) -> dict[str, Any]:
    contract = dict(audit.get("expected_contract") or {})
    baselines = dict(audit.get("backtest_hash_baselines") or {})
    criteria = dict(audit.get("acceptance_criteria") or {})
    return {
        "status": audit.get("status"),
        "validation_scope": audit.get("validation_scope"),
        "engine": contract.get("engine"),
        "shared_decision_core": contract.get("shared_decision_core") or contract.get("shared_core"),
        "fill_timing": contract.get("fill_timing"),
        "auction_mode": contract.get("auction_mode"),
        "optimized_mutations_hash": contract.get("optimized_mutations_hash"),
        "candidate_snapshot_hash": contract.get("candidate_snapshot_hash"),
        "market_bar_hash": baselines.get("market_bar_hash"),
        "non_rejection_decision_hash": baselines.get("non_rejection_decision_hash"),
        "neutral_strategy_action_hash": baselines.get("neutral_strategy_action_hash"),
        "fill_hash": baselines.get("fill_hash"),
        "trade_hash": baselines.get("trade_hash"),
        "same_bar_fill_count": baselines.get("same_bar_fill_count"),
        "acceptance_criteria": {
            key: criteria.get(key)
            for key in (
                "same_bar_fill_count",
                "forced_replay_close_count",
                "rejected_order_count",
                "end_open_position_count",
                "decision_hash_mismatch_count",
                "strategy_action_hash_mismatch_count",
                "fill_hash_mismatch_count",
                "trade_hash_mismatch_count",
            )
            if key in criteria
        },
    }


def _headline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "total_trades",
        "win_rate",
        "avg_trade_net_pct",
        "avg_mfe_capture",
        "avg_mfe_r",
        "avg_mae_r",
        "worst_fold_net",
        "median_fold_net",
        "same_bar_fill_count",
        "forced_replay_close_count",
        "rejected_order_count",
        "end_open_position_count",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _add_canonical_metric_aliases(metrics: dict[str, Any]) -> None:
    if metrics.get("net_return_pct") is None and metrics.get("broker_net_return_pct") is not None:
        metrics["net_return_pct"] = metrics["broker_net_return_pct"]
    if metrics.get("max_drawdown_pct") is None and metrics.get("broker_max_drawdown_pct") is not None:
        metrics["max_drawdown_pct"] = metrics["broker_max_drawdown_pct"]


def _hygiene_pass(metrics: dict[str, Any]) -> bool:
    return all(
        float(metrics.get(key, 0.0) or 0.0) == 0.0
        for key in (
            "same_bar_fill_count",
            "forced_replay_close_count",
            "rejected_order_count",
            "end_open_position_count",
        )
    )


def _mutation_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            delta[key] = {"from": previous.get(key), "to": current.get(key)}
    return delta


def _pct_delta(current: Any, previous: Any) -> float:
    try:
        return 100.0 * (float(current or 0.0) - float(previous or 0.0))
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _manifest_latest_round(path: Path) -> int:
    data = _read_json(path)
    return max((int(item.get("round", 0) or 0) for item in data.get("rounds", [])), default=0)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
