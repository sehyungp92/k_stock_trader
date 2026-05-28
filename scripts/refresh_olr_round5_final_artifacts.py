from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtests.auto.shared.cache_keys import build_cache_key, stable_signature
from backtests.auto.shared.types import GreedyResult
from backtests.config import load_yaml_config
from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
from backtests.strategies.olr.plugin import (
    IMMUTABLE_SCORE_COMPONENTS,
    _augment_snapshot_label_metrics,
    _augment_trade_alpha_metrics,
    _cost_policy,
    _format_olr_diagnostics,
)
from backtests.strategies.olr.replay_cache import load_olr_real_replay_bundle
from backtests.strategies.olr.runner import run_olr_backtest, snapshots_from_bundle
from strategy_olr.config import OLR_CORE_VERSION


ROUND_DIR = ROOT / "data" / "backtests" / "output" / "olr" / "round_5"
CONFIG_PATH = ROOT / "config" / "optimization" / "olr.yaml"


def main() -> None:
    optimized_path = ROUND_DIR / "optimized_config.json"
    summary_path = ROUND_DIR / "run_summary.json"
    full_path = ROUND_DIR / "round_final_full_diagnostics.json"
    results_path = ROUND_DIR / "optimized_results.json"

    optimized = _load_json(optimized_path)
    run_summary = _load_json(summary_path)
    full = _load_json(full_path)
    optimized_results = _load_json(results_path)
    config = load_yaml_config(CONFIG_PATH)
    mutations = dict(optimized["mutations"])

    replay_bundle = load_olr_real_replay_bundle(config, mutations)
    result = run_olr_backtest(config, mutations, replay_bundle=replay_bundle)
    snapshots = snapshots_from_bundle(replay_bundle)
    metrics = dict(result.metrics)
    _augment_snapshot_label_metrics(metrics, snapshots)
    _augment_trade_alpha_metrics(metrics, result.trades)
    _attach_refresh_contract(metrics, config, optimized, result, mutations)

    generated = _utc_now_iso_z()
    refreshed = _refreshed_payload(optimized, metrics, generated)
    refreshed["refresh_note"] = (
        "Official training replay refreshed under the current audit/hash contract; "
        "locked OOS holdout diagnostics were retained unchanged."
    )
    refreshed["holdout_diagnostics"] = _retained_holdout(optimized.get("holdout_diagnostics"), generated)

    train = _train_summary(metrics)
    run_summary.update(
        {
            "generated_at_utc": generated,
            "train": train,
            "oos": run_summary.get("oos", {}),
            "refresh_note": refreshed["refresh_note"],
            "source_lineage": _source_lineage(metrics),
        }
    )

    final_text = _format_round_final_diagnostics(metrics, refreshed, run_summary.get("oos", {}))
    evaluation_text = _format_round_evaluation(refreshed, train, run_summary.get("oos", {}), metrics)

    refreshed["final_diagnostics"] = _final_diagnostics_status(generated)
    full_payload = dict(refreshed)
    full_payload["final_diagnostics_text_sha256"] = stable_signature(final_text)
    optimized_results["selected"] = dict(refreshed)

    _write_json(optimized_path, refreshed)
    _write_json(full_path, full_payload)
    _write_json(summary_path, run_summary)
    _write_json(results_path, optimized_results)
    (ROUND_DIR / "round_final_diagnostics.txt").write_text(final_text, encoding="utf-8")
    (ROUND_DIR / "round_evaluation.txt").write_text(evaluation_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "strategy": "olr",
                "round": 5,
                "generated_at_utc": generated,
                "train": train,
                "source_lineage": _source_lineage(metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _attach_refresh_contract(
    metrics: dict[str, Any],
    config: dict[str, Any],
    optimized: dict[str, Any],
    result: Any,
    mutations: dict[str, Any],
) -> None:
    metrics.update(
        {
            "strategy": "olr",
            "strategy_core_version": OLR_CORE_VERSION,
            "source_fingerprint": result.source_fingerprint,
            "feature_manifest_hash": result.feature_bundle_hash,
            "candidate_snapshot_hash": result.candidate_snapshot_hash,
            "capability_level": result.capability_level,
            "holdout_excluded": True,
            "paper_live_parity_required": True,
            "paper_live_parity_status": "required_before_promotion",
            "official_performance": False,
            "promotion_status": "training_only_paper_live_pending",
            "phase_score_component_count": float(len(IMMUTABLE_SCORE_COMPONENTS)),
            "phase_score_spec_hash": build_cache_key("olr.phase_score", extra=IMMUTABLE_SCORE_COMPONENTS),
            "mutation_hash": stable_signature(mutations),
            "baseline_mutation_hash": stable_signature(config.get("initial_mutations") or {}),
        }
    )
    context = {
        "strategy": "olr",
        "config": config,
        "strategy_core_version": metrics.get("strategy_core_version"),
        "source_fingerprint": metrics.get("source_fingerprint"),
        "feature_manifest_hash": metrics.get("feature_manifest_hash"),
        "candidate_snapshot_hash": metrics.get("candidate_snapshot_hash"),
        "train_start": (optimized.get("train_window") or {}).get("date_start", ""),
        "train_end": (optimized.get("train_window") or {}).get("date_end", ""),
        "initial_equity": config.get("initial_equity", ""),
        "cost_policy": metrics.get("cost_policy") or _cost_policy(config),
        "live_parity_fill_timing": metrics.get("live_parity_fill_timing") or config.get("live_parity_fill_timing"),
        "auction_mode": optimized.get("auction_mode") or config.get("auction_mode"),
        "capability_level": metrics.get("capability_level"),
        "replay_mode": metrics.get("replay_mode"),
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
    }
    execution_contract = build_execution_contract(
        context,
        metrics,
        extra={
            "paper_live_parity_status": "required_before_promotion",
            "candidate_generation_cutoffs": {
                "daily": "row_date < trade_date",
                "stage2_intraday": "timestamp < 14:30 KST",
            },
        },
    )
    attach_official_metric_contract(
        metrics,
        primary_metric="official_mtm_net_return_pct",
        requires_audit_pass=True,
        audit_pass=True,
        audit_status="direct_official_training_replay_paper_live_pending",
        official_replay_pass=True,
        execution_contract=execution_contract,
    )


def _refreshed_payload(base: dict[str, Any], metrics: dict[str, Any], generated: str) -> dict[str, Any]:
    payload = dict(base)
    scalar_keys = (
        "audit_pass",
        "audit_status",
        "candidate_snapshot_hash",
        "capability_level",
        "decision_hash",
        "end_open_position_count",
        "entry_fill_count",
        "entry_level_trade_count",
        "entry_level_win_rate",
        "feature_manifest_hash",
        "fill_hash",
        "final_state_hash",
        "forced_replay_close_count",
        "live_parity_fill_timing",
        "max_drawdown_pct",
        "net_return_pct",
        "neutral_action_hash",
        "official_metric_basis",
        "official_mtm_max_drawdown_pct",
        "official_mtm_net_return_pct",
        "official_replay_pass",
        "paper_live_parity_status",
        "phase_score_component_count",
        "phase_score_spec_hash",
        "primary_promotion_basis",
        "primary_promotion_metric",
        "primary_promotion_value",
        "profit_factor",
        "promotion_requires_audit_pass",
        "promotion_status",
        "rejected_order_count",
        "same_bar_fill_count",
        "sharpe",
        "source_fingerprint",
        "source_snapshot_hash",
        "state_snapshot_hash",
        "strategy_core_version",
        "total_trades",
        "trade_hash",
        "win_rate",
    )
    for key in scalar_keys:
        if key in metrics:
            payload[key] = metrics[key]
    payload["generated_at_utc"] = generated
    payload["round"] = 5
    payload["strategy"] = "olr"
    payload["shared_decision_core"] = True
    payload["fill_timing"] = metrics.get("live_parity_fill_timing")
    payload["sharpe_ratio"] = metrics.get("sharpe")
    payload["primary_promotion_value"] = metrics.get("official_mtm_net_return_pct")
    payload["metric_contract"] = metrics.get("metric_contract", {})
    payload["execution_contract"] = metrics.get("execution_contract", {})
    payload["cost_policy"] = metrics.get("cost_policy", {})
    payload["score_band_attribution"] = {
        "selected_counts": metrics.get("score_band_rule_selected_counts", {}),
        "trade_counts": metrics.get("score_band_rule_trade_counts", {}),
        "realized_total_r": metrics.get("score_band_rule_realized_total_r", {}),
        "dynamic_overlay_selected_count": metrics.get("dynamic_overlay_selected_count", 0.0),
        "dynamic_overlay_trade_count": metrics.get("dynamic_overlay_trade_count", 0.0),
        "dynamic_overlay_realized_total_r": metrics.get("dynamic_overlay_realized_total_r", 0.0),
    }
    return payload


def _format_round_final_diagnostics(
    metrics: dict[str, Any],
    payload: dict[str, Any],
    oos: dict[str, Any],
) -> str:
    greedy = GreedyResult(
        base_score=0.0,
        final_score=0.0,
        final_mutations=dict(payload.get("mutations") or {}),
        kept_features=[],
        rounds=[],
        final_metrics=dict(metrics),
        total_candidates=0,
        accepted_count=0,
        elapsed_seconds=0.0,
    )
    text = _format_olr_diagnostics(5, metrics, greedy, payload.get("execution_contract", {})).rstrip()
    return "\n".join(
        [
            text,
            "",
            "## Locked OOS Holdout",
            f"- Official MTM return: {_pct(oos.get('official_mtm_net_return_pct'))}",
            f"- Entry fills: {int(_float(oos.get('entry_fill_count')))}",
            f"- Entry-level win rate: {_pct(oos.get('entry_level_win_rate'))}",
            f"- Max drawdown: {_pct(oos.get('official_mtm_max_drawdown_pct'))}",
            f"- Profit factor: {_float(oos.get('profit_factor')):.3f}",
            "- Refresh status: retained from the locked round-5 holdout artifact; not rerun during this train-artifact refresh.",
            "",
            "## Verdict",
            "Round 5 training performance is unchanged after the audit/hash refresh. Production promotion remains blocked until paper/live parity evidence is available.",
            "",
        ]
    )


def _format_round_evaluation(
    payload: dict[str, Any],
    train: dict[str, Any],
    oos: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "OLR Round 5 Evaluation Refresh",
            "=" * 30,
            f"Generated: {payload['generated_at_utc']}",
            f"Selected label: {payload.get('selected_label', '')}",
            "",
            "Training replay:",
            f"- MTM return: {_pct(train.get('official_mtm_net_return_pct'))}",
            f"- Entry fills: {int(_float(train.get('entry_fill_count')))}",
            f"- Entry win rate: {_pct(train.get('entry_level_win_rate'))}",
            f"- Max drawdown: {_pct(train.get('official_mtm_max_drawdown_pct'))}",
            f"- Profit factor: {_float(train.get('profit_factor')):.3f}",
            "",
            "Locked OOS holdout:",
            f"- MTM return: {_pct(oos.get('official_mtm_net_return_pct'))}",
            f"- Entry fills: {int(_float(oos.get('entry_fill_count')))}",
            f"- Entry win rate: {_pct(oos.get('entry_level_win_rate'))}",
            f"- Max drawdown: {_pct(oos.get('official_mtm_max_drawdown_pct'))}",
            f"- Profit factor: {_float(oos.get('profit_factor')):.3f}",
            "",
            "Audit hashes:",
            f"- Decision hash: {metrics.get('decision_hash', '')}",
            f"- Neutral action hash: {metrics.get('neutral_action_hash', '')}",
            f"- Fill hash: {metrics.get('fill_hash', '')}",
            f"- Trade hash: {metrics.get('trade_hash', '')}",
            f"- Source snapshot hash: {metrics.get('source_snapshot_hash', '')}",
            f"- Final state hash: {metrics.get('final_state_hash', '')}",
            "",
        ]
    )


def _train_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_mtm_net_return_pct": metrics.get("official_mtm_net_return_pct"),
        "entry_fill_count": metrics.get("entry_fill_count"),
        "entry_level_win_rate": metrics.get("entry_level_win_rate"),
        "official_mtm_max_drawdown_pct": metrics.get("official_mtm_max_drawdown_pct"),
        "profit_factor": metrics.get("entry_level_profit_factor", metrics.get("profit_factor")),
    }


def _retained_holdout(holdout: Any, generated: str) -> dict[str, Any]:
    payload = dict(holdout or {})
    payload["refresh_status"] = "retained_locked_holdout_artifact"
    payload["retained_at_utc"] = generated
    return payload


def _source_lineage(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_fingerprint": metrics.get("source_fingerprint"),
        "feature_manifest_hash": metrics.get("feature_manifest_hash"),
        "candidate_snapshot_hash": metrics.get("candidate_snapshot_hash"),
        "source_snapshot_hash": metrics.get("source_snapshot_hash"),
        "final_state_hash": metrics.get("final_state_hash"),
        "decision_hash": metrics.get("decision_hash"),
        "neutral_action_hash": metrics.get("neutral_action_hash"),
        "fill_hash": metrics.get("fill_hash"),
        "trade_hash": metrics.get("trade_hash"),
    }


def _final_diagnostics_status(generated: str) -> dict[str, Any]:
    return {
        "generated_at_utc": generated,
        "mode": "round5_official_train_replay_hash_contract_refresh",
        "strategy": "olr",
        "round_final_diagnostics_path": str(ROUND_DIR / "round_final_diagnostics.txt"),
        "round_final_diagnostics_exists": True,
        "round_evaluation_path": str(ROUND_DIR / "round_evaluation.txt"),
        "round_evaluation_exists": True,
        "round_final_full_diagnostics_path": str(ROUND_DIR / "round_final_full_diagnostics.json"),
        "round_final_full_diagnostics_exists": True,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pct(value: Any) -> str:
    return f"{100.0 * _float(value):.3f}%"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
