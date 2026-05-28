from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests.strategies.kalcb.fixed_trade_plan_phase import KALCBFixedTradePlanOptimizationPlugin, score_fixed


ROOT = Path(".")
ROUND_DIR = ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
OUT_DIR = ROUND_DIR / "holdout_mutation_attribution"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config" / "optimization" / "kalcb.yaml").read_text(encoding="utf-8")) or {}
    optimized = json.loads((ROUND_DIR / "optimized_config.json").read_text(encoding="utf-8"))
    state = json.loads((ROUND_DIR / "phase_state.json").read_text(encoding="utf-8"))
    final_mutations = dict(optimized["mutations"])
    baseline_mutations = dict((state["phase_results"]["1"] or {})["base_mutations"])

    holdout_config = deepcopy(config)
    baseline_window = dict(holdout_config.get("baseline") or {})
    holdout_start = str(baseline_window["holdout_start"])
    holdout_end = str(baseline_window["holdout_end"])
    holdout_config["start"] = holdout_start
    holdout_config["end"] = holdout_end
    holdout_config["use_full_available_window"] = True
    holdout_config["fixed_candidate_source"] = {
        "path": final_mutations["_kalcb.source.path"],
        "section": final_mutations["_kalcb.source.section"],
        "rank": final_mutations["_kalcb.source.rank"],
    }

    plugin = KALCBFixedTradePlanOptimizationPlugin(holdout_config, output_dir=OUT_DIR, max_workers=2)

    accepted_groups = []
    for phase_text, row in sorted((state.get("phase_results") or {}).items(), key=lambda item: int(item[0])):
        new_mutations = dict(row.get("new_mutations") or {})
        if not new_mutations:
            continue
        accepted_groups.append(
            {
                "phase": int(phase_text),
                "focus": row.get("focus"),
                "kept": list(row.get("kept_features") or []),
                "mutations": new_mutations,
            }
        )

    evaluations: list[dict[str, Any]] = []
    baseline_metrics = evaluate(plugin, "round5_starting_baseline", "baseline", baseline_mutations, None)
    final_metrics = evaluate(plugin, "round5_final_stack", "final", final_mutations, baseline_metrics["metrics"])
    evaluations.extend([baseline_metrics, final_metrics])

    cumulative = dict(baseline_mutations)
    for group in accepted_groups:
        phase = group["phase"]
        group_name = "+".join(group["kept"]) or f"phase_{phase}"

        standalone = dict(baseline_mutations)
        standalone.update(group["mutations"])
        evaluations.append(
            evaluate(
                plugin,
                f"phase{phase}_standalone_{group_name}",
                "standalone_group",
                standalone,
                baseline_metrics["metrics"],
                group=group,
            )
        )

        cumulative.update(group["mutations"])
        evaluations.append(
            evaluate(
                plugin,
                f"phase{phase}_cumulative_{group_name}",
                "cumulative_group",
                dict(cumulative),
                baseline_metrics["metrics"],
                group=group,
            )
        )

        leave_one_out = restore_keys(final_mutations, baseline_mutations, group["mutations"].keys())
        evaluations.append(
            evaluate(
                plugin,
                f"phase{phase}_leave_one_out_{group_name}",
                "leave_one_group_out",
                leave_one_out,
                final_metrics["metrics"],
                group=group,
            )
        )

    key_rows: list[dict[str, Any]] = []
    for group in accepted_groups:
        for key, value in group["mutations"].items():
            standalone = dict(baseline_mutations)
            standalone[key] = value
            key_rows.append(
                evaluate(
                    plugin,
                    f"phase{group['phase']}_standalone_key_{key}",
                    "standalone_key",
                    standalone,
                    baseline_metrics["metrics"],
                    group={"phase": group["phase"], "key": key, "value": value},
                )
            )
            leave_one_out = restore_keys(final_mutations, baseline_mutations, (key,))
            key_rows.append(
                evaluate(
                    plugin,
                    f"phase{group['phase']}_leave_one_key_out_{key}",
                    "leave_one_key_out",
                    leave_one_out,
                    final_metrics["metrics"],
                    group={"phase": group["phase"], "key": key, "value": value},
                )
            )

    payload = {
        "strategy": "kalcb",
        "round": 5,
        "window": {"start": holdout_start, "end": holdout_end, "policy": "holdout_only_no_training_dates"},
        "accepted_groups": accepted_groups,
        "baseline_label": baseline_metrics["label"],
        "final_label": final_metrics["label"],
        "evaluations": evaluations,
        "key_level_evaluations": key_rows,
        "summary": summarize(evaluations, key_rows, baseline_metrics["metrics"], final_metrics["metrics"]),
    }
    (OUT_DIR / "round5_holdout_mutation_attribution.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (OUT_DIR / "round5_holdout_mutation_attribution.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def evaluate(
    plugin: KALCBFixedTradePlanOptimizationPlugin,
    label: str,
    kind: str,
    mutations: dict[str, Any],
    compare: dict[str, Any] | None,
    *,
    group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"evaluating {label}", flush=True)
    try:
        metrics = plugin.evaluate_mutations(mutations)
    except Exception as exc:
        return {
            "label": label,
            "kind": kind,
            "group": group or {},
            "metrics": {},
            "mutation_count": len(mutations),
            "invalid": True,
            "error": f"{type(exc).__name__}: {exc}",
            "beneficial_vs_compare": False,
            "delta_vs_compare": {},
        }
    score = score_fixed(metrics)
    compact = compact_metrics(metrics, score)
    out = {
        "label": label,
        "kind": kind,
        "group": group or {},
        "metrics": compact,
        "mutation_count": len(mutations),
    }
    if compare is not None:
        out["delta_vs_compare"] = metric_delta(compact, compact_metrics(compare, score_fixed(compare)))
        out["beneficial_vs_compare"] = is_beneficial(out["delta_vs_compare"])
    return out


def restore_keys(final_mutations: dict[str, Any], baseline_mutations: dict[str, Any], keys: Any) -> dict[str, Any]:
    out = dict(final_mutations)
    for key in keys:
        if key in baseline_mutations:
            out[key] = baseline_mutations[key]
        else:
            out.pop(key, None)
    return out


def compact_metrics(metrics: dict[str, Any], score: float) -> dict[str, Any]:
    keys = [
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "broker_max_drawdown_pct",
        "trade_count",
        "avg_trade_net_pct",
        "net_win_share",
        "avg_mfe_capture",
        "mae_le_neg_1_share",
        "avg_mfe_r",
        "avg_mae_r",
        "worst_fold_net",
        "same_bar_fill_count",
        "end_open_position_count",
        "exit_reason_eod_flatten_share",
        "exit_reason_mfe_floor_share",
        "exit_reason_failed_followthrough_share",
        "exit_reason_target_r_share",
    ]
    out = {key: metrics.get(key) for key in keys if key in metrics}
    out["immutable_score"] = score
    return out


def metric_delta(metrics: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or not isinstance(compare.get(key), (int, float)):
            continue
        out[key] = float(value) - float(compare[key])
    return out


def is_beneficial(delta: dict[str, Any]) -> bool:
    return (
        float(delta.get("broker_net_return_pct", 0.0) or 0.0) > 0.0
        and float(delta.get("avg_trade_net_pct", 0.0) or 0.0) >= -0.0005
        and float(delta.get("broker_max_drawdown_pct", 0.0) or 0.0) <= 0.005
        and float(delta.get("same_bar_fill_count", 0.0) or 0.0) <= 0.0
        and float(delta.get("end_open_position_count", 0.0) or 0.0) <= 0.0
    )


def summarize(evaluations: list[dict[str, Any]], key_rows: list[dict[str, Any]], baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    standalone_groups = [row for row in evaluations if row["kind"] == "standalone_group"]
    leave_one_groups = [row for row in evaluations if row["kind"] == "leave_one_group_out"]
    cumulative_groups = [row for row in evaluations if row["kind"] == "cumulative_group"]
    return {
        "baseline": compact_metrics(baseline, score_fixed(baseline)),
        "final": compact_metrics(final, score_fixed(final)),
        "final_delta_vs_baseline": metric_delta(compact_metrics(final, score_fixed(final)), compact_metrics(baseline, score_fixed(baseline))),
        "standalone_group_beneficial_count": sum(1 for row in standalone_groups if row.get("beneficial_vs_compare")),
        "standalone_group_count": len(standalone_groups),
        "leave_one_group_helpful_count": sum(
            1 for row in leave_one_groups if float((row.get("delta_vs_compare") or {}).get("broker_net_return_pct", 0.0) or 0.0) < 0.0
        ),
        "leave_one_group_count": len(leave_one_groups),
        "cumulative_rows": [table_row(row) for row in cumulative_groups],
        "standalone_rows": [table_row(row) for row in standalone_groups],
        "leave_one_out_rows": [table_row(row) for row in leave_one_groups],
        "standalone_key_rows": [table_row(row) for row in key_rows if row["kind"] == "standalone_key"],
        "leave_one_key_rows": [table_row(row) for row in key_rows if row["kind"] == "leave_one_key_out"],
    }


def table_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    delta = row.get("delta_vs_compare") or {}
    return {
        "label": row["label"],
        "phase": (row.get("group") or {}).get("phase"),
        "beneficial": row.get("beneficial_vs_compare"),
        "net": metrics.get("broker_net_return_pct"),
        "delta_net": delta.get("broker_net_return_pct"),
        "dd": metrics.get("broker_max_drawdown_pct"),
        "delta_dd": delta.get("broker_max_drawdown_pct"),
        "trades": metrics.get("trade_count"),
        "delta_trades": delta.get("trade_count"),
        "avg_trade": metrics.get("avg_trade_net_pct"),
        "delta_avg_trade": delta.get("avg_trade_net_pct"),
        "capture": metrics.get("avg_mfe_capture"),
        "delta_capture": delta.get("avg_mfe_capture"),
        "score": metrics.get("immutable_score"),
        "delta_score": delta.get("immutable_score"),
    }


def pct(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):+.2f}%"


def num(value: Any) -> str:
    return f"{float(value or 0.0):.0f}"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# KALCB Round 5 Holdout Mutation Attribution",
        "",
        f"Window: {payload['window']['start']} to {payload['window']['end']}",
        "",
        "## Baseline To Final",
        "",
        f"- Baseline net/DD/trades/avg-trade: {pct(summary['baseline'].get('broker_net_return_pct'))} / {pct(summary['baseline'].get('broker_max_drawdown_pct'))} / {num(summary['baseline'].get('trade_count'))} / {pct(summary['baseline'].get('avg_trade_net_pct'))}",
        f"- Final net/DD/trades/avg-trade: {pct(summary['final'].get('broker_net_return_pct'))} / {pct(summary['final'].get('broker_max_drawdown_pct'))} / {num(summary['final'].get('trade_count'))} / {pct(summary['final'].get('avg_trade_net_pct'))}",
        f"- Final delta net/DD/trades/avg-trade: {pct(summary['final_delta_vs_baseline'].get('broker_net_return_pct'))} / {pct(summary['final_delta_vs_baseline'].get('broker_max_drawdown_pct'))} / {num(summary['final_delta_vs_baseline'].get('trade_count'))} / {pct(summary['final_delta_vs_baseline'].get('avg_trade_net_pct'))}",
        "",
        "## Standalone Groups Vs Round-5 Start",
        "",
        "| Phase | Beneficial | Net | Delta Net | DD | Delta DD | Trades | Avg Trade | Delta Avg Trade | Capture |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["standalone_rows"]:
        lines.append(
            f"| {row['phase']} | {row['beneficial']} | {pct(row['net'])} | {pct(row['delta_net'])} | {pct(row['dd'])} | {pct(row['delta_dd'])} | {num(row['trades'])} | {pct(row['avg_trade'])} | {pct(row['delta_avg_trade'])} | {pct(row['capture'])} |"
        )
    lines.extend(
        [
            "",
            "## Leave-One-Group-Out Vs Final",
            "",
            "| Phase | Net Without Group | Delta Vs Final | DD | Delta DD | Trades | Avg Trade | Delta Avg Trade | Capture |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["leave_one_out_rows"]:
        lines.append(
            f"| {row['phase']} | {pct(row['net'])} | {pct(row['delta_net'])} | {pct(row['dd'])} | {pct(row['delta_dd'])} | {num(row['trades'])} | {pct(row['avg_trade'])} | {pct(row['delta_avg_trade'])} | {pct(row['capture'])} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
