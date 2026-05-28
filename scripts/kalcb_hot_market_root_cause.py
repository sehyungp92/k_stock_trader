from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _mutation_key,
)


ROOT = Path(".")
ROUND4_DIR = ROOT / "data/backtests/output/kalcb/round_4"
ROUND5_DIR = ROOT / "data/backtests/output/kalcb/round_5"
OUT_DIR = ROUND5_DIR / "root_cause_hot_market"
OUT_JSON = OUT_DIR / "hot_market_root_cause_report.json"
OUT_MD = OUT_DIR / "hot_market_root_cause_report.md"


METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "broker_max_drawdown_pct",
    "trade_count",
    "active_days",
    "avg_trade_net_pct",
    "net_win_share",
    "avg_mfe_capture",
    "mae_le_neg_1_share",
    "avg_mfe_r",
    "avg_mae_r",
    "worst_fold_net",
    "candidate_pool_count",
    "initial_active_candidate_count",
    "frontier_expansion_candidate_count",
    "candidate_pool_conversion",
    "initial_active_conversion",
    "immutable_score",
)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    round4 = _read_json(ROUND4_DIR / "optimized_config.json")
    round5 = _read_json(ROUND5_DIR / "optimized_config.json")
    structural = _read_json(ROUND5_DIR / "structural_rerun" / "round5_structural_rerun_train_validation.json")
    source_path = Path(round5["mutations"][SOURCE_PATH_MUTATION])
    source_artifact = _read_json(source_path)

    config = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    config["workers"] = 2
    config["validation_gate_enabled"] = False
    config["skip_initial_baseline_eval"] = True
    source_ref = _fixed_source(round5["mutations"])

    train_config = deepcopy(config)
    train_config["fixed_candidate_source"] = source_ref

    validation_config = deepcopy(config)
    holdout = dict(validation_config.get("baseline") or {})
    validation_config["start"] = str(holdout["holdout_start"])
    validation_config["end"] = str(holdout["holdout_end"])
    validation_config["use_full_available_window"] = True
    validation_config["validation_gate_enabled"] = False
    validation_config["fixed_candidate_source"] = source_ref

    _write_progress(started, "initialising_train_plugin")
    train_plugin = KALCBFixedTradePlanOptimizationPlugin(train_config, output_dir=OUT_DIR / "train_replay", max_workers=2)
    _write_progress(started, "initialising_holdout_plugin")
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(validation_config, output_dir=OUT_DIR / "holdout_replay", max_workers=2)

    _write_progress(started, "evaluating_round5_final")
    final_mutations = dict(round5["mutations"])
    train_metrics, train_rows = _evaluate_with_rows(train_plugin, final_mutations)
    holdout_metrics, holdout_rows = _evaluate_with_rows(holdout_plugin, final_mutations)

    _write_progress(started, "evaluating_round4_baseline")
    round4_train_metrics, round4_train_rows = _evaluate_with_rows(train_plugin, dict(round4["mutations"]))
    round4_holdout_metrics, round4_holdout_rows = _evaluate_with_rows(holdout_plugin, dict(round4["mutations"]))

    _write_progress(started, "summarising_source_and_candidates")
    source_summary = _source_summary(source_artifact, source_ref)
    candidate_summary = _candidate_snapshot_summary(train_plugin.context.compiled_replay.snapshots, final_mutations)

    _write_progress(started, "summarising_trade_paths")
    trade_path = {
        "train": _trade_path_summary(train_rows),
        "holdout": _trade_path_summary(holdout_rows),
        "round4_train": _trade_path_summary(round4_train_rows),
        "round4_holdout": _trade_path_summary(round4_holdout_rows),
    }
    discrimination = {
        "train": _feature_discrimination_summary(train_rows),
        "holdout": _feature_discrimination_summary(holdout_rows),
    }

    _write_progress(started, "probing_hypotheses")
    probes = _run_hypothesis_probes(train_plugin, holdout_plugin, final_mutations, train_metrics, holdout_metrics)

    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_ref": source_ref,
        "windows": {
            "train": {
                "start": train_plugin.context.train_dates[0].isoformat(),
                "end": train_plugin.context.train_dates[-1].isoformat(),
                "sessions": len(train_plugin.context.train_dates),
            },
            "holdout": {
                "start": holdout_plugin.context.train_dates[0].isoformat(),
                "end": holdout_plugin.context.train_dates[-1].isoformat(),
                "sessions": len(holdout_plugin.context.train_dates),
            },
        },
        "round5_final": {
            "train": _compact_metrics(train_metrics),
            "holdout": _compact_metrics(holdout_metrics),
        },
        "round4_baseline": {
            "train": _compact_metrics(round4_train_metrics),
            "holdout": _compact_metrics(round4_holdout_metrics),
        },
        "round5_vs_round4_delta": {
            "train": _numeric_delta(_compact_metrics(train_metrics), _compact_metrics(round4_train_metrics)),
            "holdout": _numeric_delta(_compact_metrics(holdout_metrics), _compact_metrics(round4_holdout_metrics)),
        },
        "source_research_module": source_summary,
        "candidate_and_first30_layer": candidate_summary,
        "trade_path": trade_path,
        "feature_discrimination": discrimination,
        "hypothesis_probes": probes,
        "structural_rerun_summary": structural.get("summary", {}),
        "structural_rerun_round5_vs_round4_delta": structural.get("round5_vs_round4_delta", {}),
        "interpretation": _interpret(payload_placeholder=True),
    }
    payload["interpretation"] = _interpret(payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    OUT_MD.write_text(_markdown_report(payload), encoding="utf-8")
    _write_progress(started, "done")
    print(json.dumps(payload["interpretation"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_source(mutations: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": mutations[SOURCE_PATH_MUTATION],
        "section": mutations[SOURCE_SECTION_MUTATION],
        "rank": int(mutations[SOURCE_RANK_MUTATION]),
    }


def _evaluate_with_rows(
    plugin: KALCBFixedTradePlanOptimizationPlugin, mutations: dict[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    metrics = plugin.evaluate_mutations(mutations)
    detail = plugin._evaluation_details[_mutation_key(mutations)]
    return dict(metrics), tuple(detail.trade_rows)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def _numeric_delta(now: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in now.items():
        if isinstance(value, (int, float)) and isinstance(before.get(key), (int, float)):
            out[key] = float(value) - float(before[key])
    return out


def _source_summary(source: dict[str, Any], source_ref: dict[str, Any]) -> dict[str, Any]:
    sections = {}
    keys = (
        "portfolio_proxy_net_return_pct",
        "portfolio_proxy_max_drawdown_pct",
        "slot_cumulative_net_return_pct",
        "slot_max_drawdown_net_pct",
        "calendar_day_net_pct",
        "active_day_net_pct",
        "active_days",
        "candidate_days",
        "avg_candidates_per_session",
        "avg_mfe_r",
        "net_win_share",
        "mae_le_neg_1_share",
    )
    for section in ("top_portfolio_proxy", "top_slot_return", "top_mfe", "top_combined", "top_pareto"):
        rows = source.get(section) or []
        section_rows = []
        for idx, row in enumerate(rows[:8]):
            metrics = dict(row.get("metrics") or {})
            section_rows.append(
                {
                    "rank": idx,
                    "name": str(row.get("name") or "")[:180],
                    "combined_score": row.get("combined_score"),
                    "return_score": row.get("return_score"),
                    "mfe_score": row.get("mfe_score"),
                    "pareto_score": row.get("pareto_score"),
                    "metrics": {key: metrics.get(key) for key in keys},
                    "first30": {
                        "top_n": (row.get("first30") or {}).get("top_n"),
                        "min_first30_ret": (row.get("first30") or {}).get("min_first30_ret"),
                        "min_rel_volume": (row.get("first30") or {}).get("min_rel_volume"),
                        "min_close_location": (row.get("first30") or {}).get("min_close_location"),
                        "score_mode": (row.get("first30") or {}).get("score_mode"),
                    },
                    "frontier": {
                        "mode": (row.get("frontier") or {}).get("mode"),
                        "frontier_size": (row.get("frontier") or {}).get("frontier_size"),
                        "min_flow_5d": (row.get("frontier") or {}).get("min_flow_5d"),
                        "max_flow_divergence": (row.get("frontier") or {}).get("max_flow_divergence"),
                    },
                }
            )
        sections[section] = section_rows

    selected_rows = source.get(str(source_ref["section"])) or []
    selected = selected_rows[int(source_ref["rank"])] if selected_rows else {}
    selected_metrics = dict(selected.get("metrics") or {})
    return {
        "source_row_count": len(source.get("rows") or []),
        "selected_section": source_ref["section"],
        "selected_rank": source_ref["rank"],
        "selected_name": selected.get("name"),
        "selected_metrics": {key: selected_metrics.get(key) for key in keys},
        "top_sections": sections,
        "first30_leaderboard_top5": _first30_leaderboard_summary(source.get("first30_leaderboard") or []),
        "diagnosis": (
            "The research layer selected the rank-0 portfolio-proxy/combined/MFE/Pareto row. "
            "This is not an obviously bad source pick; its own proxy expected a high win share with low proxy MAE, "
            "but the proxy uses fixed first30 candidate outcomes and materially understates live-core path risk."
        ),
    }


def _first30_leaderboard_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows[:5]):
        metrics = dict(row.get("metrics") or {})
        spec = dict(row.get("spec") or {})
        out.append(
            {
                "rank": idx,
                "name": row.get("name"),
                "score": row.get("score"),
                "worst_fold_score": row.get("worst_fold_score"),
                "spec": {
                    "top_n": spec.get("top_n"),
                    "min_first30_ret": spec.get("min_first30_ret"),
                    "min_vwap_ret": spec.get("min_vwap_ret"),
                    "min_gap": spec.get("min_gap"),
                    "min_rel_volume": spec.get("min_rel_volume"),
                    "min_close_location": spec.get("min_close_location"),
                    "max_open_drawdown": spec.get("max_open_drawdown"),
                    "score_mode": spec.get("score_mode"),
                },
                "metrics": {
                    "slot_cumulative_net_return_pct": metrics.get("slot_cumulative_net_return_pct"),
                    "portfolio_proxy_net_return_pct": metrics.get("portfolio_proxy_net_return_pct"),
                    "active_days": metrics.get("active_days"),
                    "candidate_days": metrics.get("candidate_days"),
                    "net_win_share": metrics.get("net_win_share"),
                    "avg_mfe_r": metrics.get("avg_mfe_r"),
                    "mae_le_neg_1_share": metrics.get("mae_le_neg_1_share"),
                },
            }
        )
    return out


def _candidate_snapshot_summary(snapshots: dict[Any, Any], mutations: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for day, snapshot in snapshots.items():
        active = {str(symbol) for symbol in ((snapshot.metadata or {}).get("active_symbols") or [])}
        for cand in snapshot.candidates:
            metadata = dict(cand.metadata or {})
            first30_ret = _num(metadata.get("first30_ret"))
            first30_vwap_ret = _num(metadata.get("first30_vwap_ret"))
            first30_rel_volume = _num(metadata.get("first30_rel_volume"))
            first30_cpr = _num(metadata.get("first30_signal_bar_cpr", metadata.get("first30_close_location")))
            first30_range_atr = _num(metadata.get("first30_range_atr"))
            frontier_rank = int(metadata.get("frontier_rank") or 0)
            quality_votes = 0
            quality_votes += first30_ret >= float(mutations.get("kalcb.entry.quality_min_bar_ret", 0.01) or 0.01)
            quality_votes += first30_cpr >= float(mutations.get("kalcb.entry.quality_min_first30_signal_cpr", 0.75) or 0.75)
            quality_votes += first30_rel_volume >= float(mutations.get("kalcb.entry.quality_min_first30_rel_volume", 2.0) or 2.0)
            quality_votes += first30_range_atr >= float(mutations.get("kalcb.entry.quality_min_first30_range_atr", 0.75) or 0.75)
            quality_votes += float(getattr(cand, "flow_score", 0.0) or 0.0) >= float(mutations.get("kalcb.entry.quality_min_flow_score", -0.05) or -0.05)
            quality_votes += float(getattr(cand, "accumulation_score", 0.0) or 0.0) >= float(
                mutations.get("kalcb.entry.quality_min_accumulation_score", 0.0) or 0.0
            )
            quality_votes += frontier_rank <= int(mutations.get("kalcb.entry.quality_max_frontier_rank", 20) or 20)
            current_gate_no_initial = (
                frontier_rank <= int(mutations.get("kalcb.entry.max_frontier_rank", 12) or 12)
                and first30_ret >= float(mutations.get("kalcb.entry.min_bar_ret", 0.01) or 0.01)
                and first30_vwap_ret >= float(mutations.get("kalcb.entry.min_vwap_ret", 0.0) or 0.0)
                and first30_rel_volume >= float(mutations.get("kalcb.entry.min_first30_rel_volume", 1.0) or 1.0)
                and quality_votes >= int(mutations.get("kalcb.entry.min_quality_votes", 6) or 6)
            )
            rows.append(
                {
                    "day": str(day),
                    "symbol": cand.symbol,
                    "active": cand.symbol in active,
                    "frontier_rank": frontier_rank,
                    "candidate_rank": int(metadata.get("candidate_rank") or 0),
                    "first30_ret": first30_ret,
                    "first30_vwap_ret": first30_vwap_ret,
                    "first30_rel_volume": first30_rel_volume,
                    "first30_close_location": _num(metadata.get("first30_close_location")),
                    "first30_cpr": first30_cpr,
                    "first30_range_atr": first30_range_atr,
                    "flow_score": float(getattr(cand, "flow_score", 0.0) or 0.0),
                    "accumulation_score": float(getattr(cand, "accumulation_score", 0.0) or 0.0),
                    "quality_votes": quality_votes,
                    "current_gate_no_initial": bool(current_gate_no_initial),
                }
            )

    active_rows = [row for row in rows if row["active"]]
    inactive_rows = [row for row in rows if not row["active"]]
    gated_rows = [row for row in rows if row["current_gate_no_initial"]]
    gated_inactive = [row for row in gated_rows if not row["active"]]
    by_rank = _bucket_summary(rows, lambda r: _rank_bucket(r["frontier_rank"]), value_keys=("first30_ret", "first30_rel_volume", "quality_votes"))
    return {
        "candidate_pool_rows": len(rows),
        "initial_active_rows": len(active_rows),
        "frontier_shadow_rows": len(inactive_rows),
        "current_gate_pass_without_initial_active": len(gated_rows),
        "current_gate_pass_but_blocked_by_initial_active": len(gated_inactive),
        "active_first30": _feature_means(active_rows),
        "inactive_first30": _feature_means(inactive_rows),
        "blocked_current_gate_first30": _feature_means(gated_inactive),
        "frontier_rank_buckets": by_rank,
        "diagnosis": (
            "The first30 layer is restrictive mainly because the live route only acts on initial_active symbols. "
            "There are many non-active candidates that pass the same visible first30/quality gates, but earlier replay probes show "
            "that opening this channel broadly does not survive train/holdout validation."
        ),
    }


def _feature_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "frontier_rank",
        "candidate_rank",
        "first30_ret",
        "first30_vwap_ret",
        "first30_rel_volume",
        "first30_close_location",
        "first30_cpr",
        "first30_range_atr",
        "flow_score",
        "accumulation_score",
        "quality_votes",
    )
    out = {"count": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        out[f"avg_{key}"] = _mean(vals)
        out[f"median_{key}"] = _median(vals)
    return out


def _bucket_summary(rows: list[dict[str, Any]], bucket_fn: Any, value_keys: tuple[str, ...]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(bucket_fn(row))].append(row)
    out = {}
    for bucket, items in sorted(buckets.items()):
        summary = {"count": len(items)}
        for key in value_keys:
            values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
            summary[f"avg_{key}"] = _mean(values)
        summary["active_share"] = sum(1 for item in items if item.get("active")) / max(len(items), 1)
        summary["current_gate_no_initial_share"] = sum(1 for item in items if item.get("current_gate_no_initial")) / max(len(items), 1)
        out[bucket] = summary
    return out


def _trade_path_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    total_r = sum(_num(row.get("r")) for row in rows)
    pos_mfe = [max(_num(row.get("mfe_r")), 0.0) for row in rows]
    losers = [row for row in rows if _num(row.get("r")) < 0]
    winners = [row for row in rows if _num(row.get("r")) > 0]
    exit_reasons = {}
    for reason, items in _group(rows, lambda r: str(r.get("exit_reason") or "unknown")).items():
        exit_reasons[reason] = _r_summary(items)
    return {
        "count": len(rows),
        "total_r": total_r,
        "avg_r": total_r / max(len(rows), 1),
        "win_share": len(winners) / max(len(rows), 1),
        "avg_mfe_r": _mean(pos_mfe),
        "median_mfe_r": _median(pos_mfe),
        "mfe_ge_1_share": sum(1 for value in pos_mfe if value >= 1.0) / max(len(rows), 1),
        "mfe_ge_5_share": sum(1 for value in pos_mfe if value >= 5.0) / max(len(rows), 1),
        "mfe_ge_10_share": sum(1 for value in pos_mfe if value >= 10.0) / max(len(rows), 1),
        "avg_mae_r": _mean([_num(row.get("mae_r")) for row in rows]),
        "mae_le_neg1_share": sum(1 for row in rows if _num(row.get("mae_r")) <= -1.0) / max(len(rows), 1),
        "actual_mfe_capture": sum(max(_num(row.get("r")), 0.0) for row in rows) / max(sum(pos_mfe), 1e-9),
        "avg_giveback_r": _mean([_num(row.get("giveback_r")) for row in rows]),
        "loser_count": len(losers),
        "loser_share": len(losers) / max(len(rows), 1),
        "loser_avg_mae_r": _mean([_num(row.get("mae_r")) for row in losers]),
        "loser_avg_mfe_r": _mean([_num(row.get("mfe_r")) for row in losers]),
        "loser_mfe_ge_1_share": sum(1 for row in losers if _num(row.get("mfe_r")) >= 1.0) / max(len(losers), 1),
        "loser_mfe_ge_5_share": sum(1 for row in losers if _num(row.get("mfe_r")) >= 5.0) / max(len(losers), 1),
        "loser_mfe_ge_10_share": sum(1 for row in losers if _num(row.get("mfe_r")) >= 10.0) / max(len(losers), 1),
        "winner_avg_giveback_r": _mean([_num(row.get("giveback_r")) for row in winners]),
        "mfe_oracle_total_r_50pct_positive_mfe": sum(0.5 * value for value in pos_mfe if value > 0),
        "mfe_oracle_total_r_75pct_positive_mfe": sum(0.75 * value for value in pos_mfe if value > 0),
        "mfe_oracle_50pct_delta_r": sum(0.5 * value for value in pos_mfe if value > 0) - total_r,
        "mfe_oracle_75pct_delta_r": sum(0.75 * value for value in pos_mfe if value > 0) - total_r,
        "exit_reasons": exit_reasons,
        "frontier_rank_buckets": {bucket: _r_summary(items) for bucket, items in _group(rows, lambda r: _rank_bucket(_num(r.get("frontier_rank")))).items()},
        "first30_ret_buckets": {bucket: _r_summary(items) for bucket, items in _group(rows, lambda r: _ret_bucket(_num(r.get("first30_ret")))).items()},
        "relvol_buckets": {bucket: _r_summary(items) for bucket, items in _group(rows, lambda r: _relvol_bucket(_num(r.get("first30_rel_volume")))).items()},
        "top_lost_alpha": sorted(
            [
                {
                    "entry_date": row.get("entry_date"),
                    "symbol": row.get("symbol"),
                    "r": _num(row.get("r")),
                    "mfe_r": _num(row.get("mfe_r")),
                    "mae_r": _num(row.get("mae_r")),
                    "giveback_r": _num(row.get("giveback_r")),
                    "exit_reason": row.get("exit_reason"),
                    "frontier_rank": row.get("frontier_rank"),
                    "first30_ret": row.get("first30_ret"),
                    "first30_rel_volume": row.get("first30_rel_volume"),
                }
                for row in rows
            ],
            key=lambda item: item["giveback_r"],
            reverse=True,
        )[:12],
    }


def _feature_discrimination_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    features = (
        "frontier_rank",
        "candidate_rank",
        "frontier_selection_score",
        "flow_score",
        "accumulation_score",
        "first30_ret",
        "first30_vwap_ret",
        "first30_rel_volume",
        "first30_range_close_location",
        "first30_signal_bar_cpr",
        "first30_low_vs_prev_close",
        "first30_range_atr",
        "bar_rvol",
        "cpr",
    )
    winners = [row for row in rows if _num(row.get("r")) > 0]
    losers = [row for row in rows if _num(row.get("r")) < 0]
    out = {"count": len(rows), "features": {}}
    r_values = [_num(row.get("r")) for row in rows]
    for feature in features:
        values = [_num(row.get(feature)) for row in rows]
        valid = [(v, r) for v, r in zip(values, r_values) if math.isfinite(v) and math.isfinite(r)]
        if not valid:
            continue
        out["features"][feature] = {
            "winner_avg": _mean([_num(row.get(feature)) for row in winners]),
            "loser_avg": _mean([_num(row.get(feature)) for row in losers]),
            "winner_minus_loser": _mean([_num(row.get(feature)) for row in winners]) - _mean([_num(row.get(feature)) for row in losers]),
            "pearson_corr_to_r": _pearson([v for v, _ in valid], [r for _, r in valid]),
            "quartiles": _quartiles(rows, feature),
        }
    return out


def _run_hypothesis_probes(
    train_plugin: KALCBFixedTradePlanOptimizationPlugin,
    holdout_plugin: KALCBFixedTradePlanOptimizationPlugin,
    base_mutations: dict[str, Any],
    base_train: dict[str, Any],
    base_holdout: dict[str, Any],
) -> dict[str, Any]:
    experiments: dict[str, dict[str, Any]] = {
        "frontier_branch_rank12_current_gates": {
            "kalcb.entry.frontier_branch_universe": True,
            "kalcb.entry.require_initial_active": False,
            "kalcb.entry.max_frontier_rank": 12,
        },
        "frontier_branch_rank20_quality7": {
            "kalcb.entry.frontier_branch_universe": True,
            "kalcb.entry.require_initial_active": False,
            "kalcb.entry.max_frontier_rank": 20,
            "kalcb.entry.min_quality_votes": 7,
        },
        "frontier_branch_rank30_high_first30": {
            "kalcb.entry.frontier_branch_universe": True,
            "kalcb.entry.require_initial_active": False,
            "kalcb.entry.max_frontier_rank": 30,
            "kalcb.entry.min_bar_ret": 0.015,
            "kalcb.entry.min_vwap_ret": 0.005,
            "kalcb.entry.min_first30_rel_volume": 2.5,
            "kalcb.entry.min_quality_votes": 6,
        },
        "mfe_giveback_10_gap5_h24": {
            "kalcb.exit.mfe_giveback_enabled": True,
            "kalcb.exit.mfe_giveback_start_r": 10.0,
            "kalcb.exit.mfe_giveback_gap_r": 5.0,
            "kalcb.exit.mfe_giveback_min_hold_bars": 24,
        },
        "target_50r": {
            "kalcb.exit.target_r": 50.0,
        },
        "failed_followthrough_softer": {
            "kalcb.exit.failed_followthrough_bars": 8,
            "kalcb.exit.failed_followthrough_close_r": -0.50,
            "kalcb.exit.failed_followthrough_mfe_r": 1.0,
        },
    }
    out = {}
    for name, mutation in experiments.items():
        _write_progress(time.time(), f"probe_{name}")
        candidate_mutations = dict(base_mutations)
        candidate_mutations.update(mutation)
        try:
            train_metrics, train_rows = _evaluate_with_rows(train_plugin, candidate_mutations)
            holdout_metrics, holdout_rows = _evaluate_with_rows(holdout_plugin, candidate_mutations)
            out[name] = {
                "mutation_hash": stable_signature(mutation),
                "mutations": mutation,
                "train": _compact_metrics(train_metrics),
                "holdout": _compact_metrics(holdout_metrics),
                "train_delta": _numeric_delta(_compact_metrics(train_metrics), _compact_metrics(base_train)),
                "holdout_delta": _numeric_delta(_compact_metrics(holdout_metrics), _compact_metrics(base_holdout)),
                "train_path": _trade_path_summary(train_rows),
                "holdout_path": _trade_path_summary(holdout_rows),
            }
        except Exception as exc:
            out[name] = {"mutations": mutation, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _r_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r_values = [_num(row.get("r")) for row in rows]
    return {
        "count": len(rows),
        "total_r": sum(r_values),
        "avg_r": _mean(r_values),
        "win_share": sum(1 for value in r_values if value > 0) / max(len(r_values), 1),
        "avg_mfe_r": _mean([_num(row.get("mfe_r")) for row in rows]),
        "avg_mae_r": _mean([_num(row.get("mae_r")) for row in rows]),
        "avg_giveback_r": _mean([_num(row.get("giveback_r")) for row in rows]),
        "avg_capture": sum(max(value, 0.0) for value in r_values) / max(sum(max(_num(row.get("mfe_r")), 0.0) for row in rows), 1e-9),
    }


def _quartiles(rows: list[dict[str, Any]], feature: str) -> list[dict[str, Any]]:
    pairs = [(row, _num(row.get(feature))) for row in rows]
    pairs = [(row, value) for row, value in pairs if math.isfinite(value)]
    pairs.sort(key=lambda item: item[1])
    if not pairs:
        return []
    out = []
    for idx in range(4):
        lo = int(len(pairs) * idx / 4)
        hi = int(len(pairs) * (idx + 1) / 4)
        items = [row for row, _ in pairs[lo:hi]]
        values = [value for _, value in pairs[lo:hi]]
        out.append(
            {
                "quartile": idx + 1,
                "feature_min": min(values) if values else None,
                "feature_max": max(values) if values else None,
                **_r_summary(items),
            }
        )
    return out


def _group(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return dict(groups)


def _rank_bucket(value: float) -> str:
    value = int(value or 0)
    if value <= 0:
        return "unknown"
    if value == 1:
        return "rank_1"
    if value <= 3:
        return "rank_2_3"
    if value <= 5:
        return "rank_4_5"
    if value <= 10:
        return "rank_6_10"
    if value <= 30:
        return "rank_11_30"
    return "rank_gt_30"


def _ret_bucket(value: float) -> str:
    if value < 0.005:
        return "ret_lt_0p5"
    if value < 0.015:
        return "ret_0p5_1p5"
    return "ret_ge_1p5"


def _relvol_bucket(value: float) -> str:
    if value < 1.5:
        return "rv_lt_1p5"
    if value < 2.5:
        return "rv_1p5_2p5"
    return "rv_ge_2p5"


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def _interpret(payload: Any = None, *, payload_placeholder: bool = False) -> dict[str, Any]:
    if payload_placeholder:
        return {}
    train = payload["round5_final"]["train"]
    holdout = payload["round5_final"]["holdout"]
    train_path = payload["trade_path"]["train"]
    holdout_path = payload["trade_path"]["holdout"]
    candidate = payload["candidate_and_first30_layer"]
    probes = payload["hypothesis_probes"]
    frontier_probe = probes.get("frontier_branch_rank12_current_gates", {})
    management_probe = probes.get("mfe_giveback_10_gap5_h24", {})
    return {
        "root_cause": (
            "Alpha is present, but the current stack is not converting hot-market intraday excursion into realised return. "
            "The bottleneck is a combination of narrow opportunity routing, weak ex-ante discrimination, and path-dependent trade management leakage. "
            "The source row is strong; the first30 layer confirms movement potential; the live-core entry/exit stack then accepts too many painful paths and gives back too much MFE."
        ),
        "alpha_not_absent_evidence": {
            "train_avg_mfe_r": train.get("avg_mfe_r"),
            "train_mfe_ge_1_share": train_path.get("mfe_ge_1_share"),
            "train_oracle_50pct_mfe_delta_r": train_path.get("mfe_oracle_50pct_delta_r"),
            "holdout_avg_mfe_r": holdout.get("avg_mfe_r"),
            "holdout_mfe_ge_1_share": holdout_path.get("mfe_ge_1_share"),
            "holdout_oracle_50pct_mfe_delta_r": holdout_path.get("mfe_oracle_50pct_delta_r"),
        },
        "discrimination_problem_evidence": {
            "train_loser_share": train_path.get("loser_share"),
            "train_loser_avg_mae_r": train_path.get("loser_avg_mae_r"),
            "train_loser_mfe_ge_1_share": train_path.get("loser_mfe_ge_1_share"),
            "holdout_loser_share": holdout_path.get("loser_share"),
            "holdout_loser_avg_mae_r": holdout_path.get("loser_avg_mae_r"),
            "holdout_loser_mfe_ge_1_share": holdout_path.get("loser_mfe_ge_1_share"),
        },
        "opportunity_routing_evidence": {
            "candidate_pool_rows": candidate.get("candidate_pool_rows"),
            "initial_active_rows": candidate.get("initial_active_rows"),
            "current_gate_pass_but_blocked_by_initial_active": candidate.get("current_gate_pass_but_blocked_by_initial_active"),
            "frontier_rank12_probe_train_delta": frontier_probe.get("train_delta"),
            "frontier_rank12_probe_holdout_delta": frontier_probe.get("holdout_delta"),
        },
        "trade_management_problem_evidence": {
            "train_actual_mfe_capture": train_path.get("actual_mfe_capture"),
            "train_avg_giveback_r": train_path.get("avg_giveback_r"),
            "train_winner_avg_giveback_r": train_path.get("winner_avg_giveback_r"),
            "holdout_actual_mfe_capture": holdout_path.get("actual_mfe_capture"),
            "holdout_avg_giveback_r": holdout_path.get("avg_giveback_r"),
            "mfe_giveback_probe_train_delta": management_probe.get("train_delta"),
            "mfe_giveback_probe_holdout_delta": management_probe.get("holdout_delta"),
        },
        "not_a_simple_exit_bug": (
            "The exit mechanism fires and preserves the biggest EOD winners; the failure is policy/conditioning. "
            "Generic giveback/target mutations either cut tail winners or do not improve holdout enough, so the needed change is a path/cohort model rather than another static target."
        ),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    interp = payload["interpretation"]
    train = payload["round5_final"]["train"]
    holdout = payload["round5_final"]["holdout"]
    path = payload["trade_path"]["train"]
    hpath = payload["trade_path"]["holdout"]
    candidate = payload["candidate_and_first30_layer"]
    source = payload["source_research_module"]
    lines = [
        "# KALCB hot-market root-cause audit",
        "",
        f"Generated in {payload['elapsed_seconds']}s.",
        "",
        "## Verdict",
        "",
        interp["root_cause"],
        "",
        "## Headline metrics",
        "",
        "| Window | Net | DD | Trades | Win | Avg MFE R | MFE capture | MAE<=-1R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Train | {_pct(train.get('broker_net_return_pct'))} | {_pct(train.get('broker_max_drawdown_pct'))} | {_n(train.get('trade_count'))} | {_pct(train.get('net_win_share'))} | {_f(train.get('avg_mfe_r'))} | {_pct(train.get('avg_mfe_capture'))} | {_pct(train.get('mae_le_neg_1_share'))} |",
        f"| Holdout | {_pct(holdout.get('broker_net_return_pct'))} | {_pct(holdout.get('broker_max_drawdown_pct'))} | {_n(holdout.get('trade_count'))} | {_pct(holdout.get('net_win_share'))} | {_f(holdout.get('avg_mfe_r'))} | {_pct(holdout.get('avg_mfe_capture'))} | {_pct(holdout.get('mae_le_neg_1_share'))} |",
        "",
        "## Research and first30 layer",
        "",
        f"- Source row: rank {source['selected_rank']} in `{source['selected_section']}`.",
        f"- Source proxy: portfolio net {_pct(source['selected_metrics'].get('portfolio_proxy_net_return_pct'))}, proxy DD {_pct(abs(_num(source['selected_metrics'].get('portfolio_proxy_max_drawdown_pct'))))}, proxy win {_pct(source['selected_metrics'].get('net_win_share'))}, proxy MAE<=-1R {_pct(source['selected_metrics'].get('mae_le_neg_1_share'))}.",
        f"- Candidate pool: {_n(candidate['candidate_pool_rows'])} full candidates, {_n(candidate['initial_active_rows'])} initial-active, {_n(candidate['current_gate_pass_but_blocked_by_initial_active'])} non-active candidates pass the visible current first30/quality gates before the initial-active block.",
        "",
        "## Path evidence",
        "",
        f"- Train realised {path['total_r']:.1f}R from avg MFE {path['avg_mfe_r']:.1f}R; a naive 50% positive-MFE oracle would add {path['mfe_oracle_50pct_delta_r']:.1f}R.",
        f"- Holdout realised {hpath['total_r']:.1f}R from avg MFE {hpath['avg_mfe_r']:.1f}R; a naive 50% positive-MFE oracle would add {hpath['mfe_oracle_50pct_delta_r']:.1f}R.",
        f"- Accepted losers: train {_pct(path['loser_share'])} of trades, avg loser MAE {path['loser_avg_mae_r']:.1f}R; holdout {_pct(hpath['loser_share'])}, avg loser MAE {hpath['loser_avg_mae_r']:.1f}R.",
        "",
        "## Probe results",
        "",
        "| Probe | Train net delta | Holdout net delta | Train trades delta | Holdout trades delta | Train DD delta | Holdout DD delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, probe in payload["hypothesis_probes"].items():
        if "error" in probe:
            lines.append(f"| {name} | error | error | error | error | error | error |")
            continue
        td = probe["train_delta"]
        hd = probe["holdout_delta"]
        lines.append(
            f"| {name} | {_pct(td.get('broker_net_return_pct'))} | {_pct(hd.get('broker_net_return_pct'))} | {_f(td.get('trade_count'), 0)} | {_f(hd.get('trade_count'), 0)} | {_pct(td.get('broker_max_drawdown_pct'))} | {_pct(hd.get('broker_max_drawdown_pct'))} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            interp["not_a_simple_exit_bug"],
        ]
    )
    return "\n".join(lines) + "\n"


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _f(value: Any, places: int = 2) -> str:
    return f"{_num(value):.{places}f}"


def _n(value: Any) -> str:
    return str(int(round(_num(value))))


def _write_progress(started: float, stage: str) -> None:
    payload = {"stage": stage, "elapsed_seconds": round(time.time() - started, 3), "updated_at_epoch": time.time()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
