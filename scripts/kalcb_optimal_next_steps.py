from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kalcb_local_minimum_recovery import (  # noqa: E402
    CONFIG_PATH,
    DEFAULT_OUT_DIR as LOCAL_MINIMUM_DIR,
    GAP_RETENTION_TRAIN_THRESHOLDS,
    RecoveryEvaluator,
    _clean_metrics,
    _load_round_mutations,
    _mutation_key,
    _num,
    _round_source_ref,
    _shadow_route,
)


ROUND_ROOT = REPO_ROOT / "data" / "backtests" / "output" / "kalcb" / "round_5"
PIPELINE_DIR = ROUND_ROOT / "positive_r_pipeline_breakdown"
OUT_DIR = ROUND_ROOT / "optimal_next_steps"
SEED_PATH = LOCAL_MINIMUM_DIR / "07_alpha_conversion_next_round" / "next_round_seed_auto_pullback_q85_rank8_r0p015.json"
CHALLENGER_PATH = (
    LOCAL_MINIMUM_DIR
    / "07_alpha_conversion_next_round"
    / "next_round_challenger_auto_pullback_q85_rank8_r0p02_target60.json"
)
CANDIDATE_FEATURES = LOCAL_MINIMUM_DIR / "08_candidate_surfacing_recovery" / "candidate_surfacing_train_features.jsonl"


METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "broker_max_drawdown_pct",
    "trade_count",
    "active_days",
    "candidate_pool_count",
    "full_candidate_pool_count",
    "initial_active_candidate_count",
    "frontier_expansion_candidate_count",
    "static_route_eligible_count",
    "candidate_pool_conversion",
    "initial_active_conversion",
    "avg_trade_net_pct",
    "avg_mfe_capture",
    "avg_mfe_r",
    "avg_mae_r",
    "mae_le_neg_1_share",
    "worst_fold_net",
    "median_fold_net",
    "same_bar_fill_count",
    "end_open_position_count",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def quantile(values: list[float], pct: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    pos = (len(clean) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def feature_quantiles() -> dict[str, dict[str, float]]:
    keys = (
        "first30_rel_volume",
        "first30_signal_bar_cpr",
        "first30_gap",
        "first30_vwap_ret",
        "first30_ret",
        "daily_close20_loc",
        "daily_momentum_pct",
        "sector_daily_score_pct",
    )
    values: dict[str, list[float]] = {key: [] for key in keys}
    if not CANDIDATE_FEATURES.exists():
        return {}
    with CANDIDATE_FEATURES.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for key in keys:
                try:
                    value = float(row.get(key) or 0.0)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values[key].append(value)
    return {
        key: {
            "q50": quantile(items, 0.50),
            "q75": quantile(items, 0.75),
            "q80": quantile(items, 0.80),
            "q85": quantile(items, 0.85),
            "q90": quantile(items, 0.90),
        }
        for key, items in values.items()
    }


def source_row_digest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = list(payload.get("rows") or [])
    top: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metrics = dict(row.get("metrics") or {})
        frontier = dict(row.get("frontier") or {})
        first30 = dict(row.get("first30") or {})
        top.append(
            {
                "section": "rows",
                "rank": index,
                "name": row.get("name"),
                "full_first30_candidate_recall": metrics.get("full_first30_candidate_recall"),
                "portfolio_proxy_net_return_pct": metrics.get("portfolio_proxy_net_return_pct"),
                "portfolio_proxy_max_drawdown_pct": metrics.get("portfolio_proxy_max_drawdown_pct"),
                "candidate_days": metrics.get("candidate_days"),
                "frontier_size": frontier.get("frontier_size"),
                "frontier_min_flow_5d": frontier.get("min_flow_5d"),
                "frontier_max_flow_divergence": frontier.get("max_flow_divergence"),
                "frontier_min_adv20_krw": frontier.get("min_adv20_krw"),
                "first30_min_ret": first30.get("min_first30_ret"),
                "first30_min_gap": first30.get("min_gap"),
                "first30_min_vwap_ret": first30.get("min_vwap_ret"),
                "first30_top_n": first30.get("top_n"),
            }
        )
    top_recall = sorted(top, key=lambda row: _num(row.get("full_first30_candidate_recall")), reverse=True)[:8]
    top_net_recall = sorted(
        [row for row in top if _num(row.get("full_first30_candidate_recall")) >= 0.25],
        key=lambda row: _num(row.get("portfolio_proxy_net_return_pct")),
        reverse=True,
    )[:8]
    return {
        "source_path": str(path),
        "top_recall_rows": top_recall,
        "top_net_with_recall_ge_25pct": top_net_recall,
    }


def load_seed_mutations(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    mutations = payload.get("mutations")
    if not isinstance(mutations, dict):
        raise ValueError(f"Missing mutations in {path}")
    return copy.deepcopy(mutations)


def clone(mutations: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(mutations)


def non_anchor_routes(mutations: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        route
        for route in list(mutations.get("kalcb.entry.routes") or [])
        if isinstance(route, dict) and str(route.get("mode") or "") != "first30_open"
    ]


def mutate_first_non_anchor_route(mutations: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = clone(mutations)
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        if str(route.get("mode") or "") != "first30_open":
            route.update(copy.deepcopy(updates))
            break
    out["kalcb.entry.routes"] = routes
    return out


def set_first_non_anchor_context_min(mutations: dict[str, Any], context_min: dict[str, float]) -> dict[str, Any]:
    out = clone(mutations)
    routes = [dict(route) for route in out.get("kalcb.entry.routes") or [] if isinstance(route, dict)]
    for route in routes:
        if str(route.get("mode") or "") != "first30_open":
            existing = dict(route.get("context_min") or {})
            existing.update({key: float(value) for key, value in context_min.items()})
            route["context_min"] = existing
            break
    out["kalcb.entry.routes"] = routes
    return out


def replace_non_anchor_routes(mutations: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
    out = clone(mutations)
    anchors = [
        dict(route)
        for route in out.get("kalcb.entry.routes") or []
        if isinstance(route, dict) and str(route.get("mode") or "") == "first30_open"
    ]
    out["kalcb.entry.routes"] = [copy.deepcopy(route) for route in routes] + anchors
    return out


def experiment(label: str, family: str, purpose: str, mutations: dict[str, Any]) -> dict[str, Any]:
    return {"label": label, "family": family, "purpose": purpose, "mutations": mutations}


def build_candidates(
    base: dict[str, Any],
    seed: dict[str, Any],
    challenger: dict[str, Any],
    quantiles: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    relvol_q80 = quantiles.get("first30_rel_volume", {}).get("q80") or 3.2093901870328034
    relvol_q85 = quantiles.get("first30_rel_volume", {}).get("q85") or 3.7539582388352253
    relvol_q90 = quantiles.get("first30_rel_volume", {}).get("q90") or 4.696757900023581
    sector_q75 = GAP_RETENTION_TRAIN_THRESHOLDS.get("sector_daily_score_pct_q75", 81.25)
    source_path = str(seed.get("_kalcb.source.path") or base.get("_kalcb.source.path") or "")

    pullback_route = non_anchor_routes(seed)[0] if non_anchor_routes(seed) else {}
    avwap_015 = _shadow_route(
        name="optimal_avwap_rank8_r0p015",
        mode="avwap_reclaim",
        rank=8,
        risk_mult=0.015,
        max_session_trades=1,
        context_min={"first30_rel_volume": GAP_RETENTION_TRAIN_THRESHOLDS["first30_rel_volume_q85"]},
    )
    avwap_020_target60 = _shadow_route(
        name="optimal_avwap_rank8_r0p02_target60",
        mode="avwap_reclaim",
        rank=8,
        risk_mult=0.02,
        max_session_trades=1,
        context_min={"first30_rel_volume": GAP_RETENTION_TRAIN_THRESHOLDS["first30_rel_volume_q85"]},
    )
    pullback_015 = dict(pullback_route)
    avwap_015["priority"] = 1

    candidates = [
        experiment("round5_current", "baseline_control", "Current round-5 optimized mutation stack.", clone(base)),
        experiment("seed_pullback_r0p015", "route_seed", "Conservative next-round pullback route seed from local-minimum recovery.", clone(seed)),
        experiment("challenger_pullback_r0p02_target60", "route_target", "Upside challenger: pullback route at 0.02 risk with target-60 overlay.", clone(challenger)),
        experiment(
            "seed_relvol_q90_feature",
            "route_breadth",
            "Relax delayed-route relvol context from gap-retention q85 to feature q90.",
            set_first_non_anchor_context_min(seed, {"first30_rel_volume": relvol_q90}),
        ),
        experiment(
            "seed_relvol_q85_feature",
            "route_breadth",
            "Relax delayed-route relvol context to feature q85; tests whether the q85 route gate is too tight.",
            set_first_non_anchor_context_min(seed, {"first30_rel_volume": relvol_q85}),
        ),
        experiment(
            "seed_relvol_q80_feature",
            "route_breadth",
            "More aggressive route breadth probe using feature q80 first30 RVOL.",
            set_first_non_anchor_context_min(seed, {"first30_rel_volume": relvol_q80}),
        ),
        experiment(
            "seed_relvol_q85_sector_q75",
            "route_breadth_sector",
            "Broaden route relvol but require strong daily sector score to offset added noise.",
            set_first_non_anchor_context_min(seed, {"first30_rel_volume": relvol_q85, "sector_daily_score_pct": sector_q75}),
        ),
        experiment(
            "seed_quality_votes5_cpr65",
            "entry_gate",
            "Soften quality-vote and CPR requirements where selected-to-entered audit showed avoidable R cuts.",
            mutate_first_non_anchor_route(
                {**clone(seed), "kalcb.entry.min_quality_votes": 5, "kalcb.entry.quality_min_first30_signal_cpr": 0.65},
                {"min_quality_votes": 5, "quality_min_first30_signal_cpr": 0.65},
            ),
        ),
        experiment(
            "seed_entry_minbar_005_qualitybar_005",
            "entry_gate",
            "Lower first30/min-quality bar return from 1.0% to 0.5%.",
            {
                **clone(seed),
                "kalcb.entry.min_bar_ret": 0.005,
                "kalcb.entry.quality_min_bar_ret": 0.005,
            },
        ),
        experiment(
            "seed_entry_minbar_000_qualitybar_000",
            "entry_gate",
            "Remove the first30/min-quality bar return floor to test the largest selected-to-entered blocker.",
            {
                **clone(seed),
                "kalcb.entry.min_bar_ret": 0.0,
                "kalcb.entry.quality_min_bar_ret": 0.0,
            },
        ),
        experiment(
            "seed_source_rows_rank4_recall_top12",
            "source_candidate_surfacing",
            "Use the broader source row with higher full-first30 recall and top12 first30 selector.",
            {
                **clone(seed),
                "_kalcb.source.path": source_path,
                "_kalcb.source.section": "rows",
                "_kalcb.source.rank": 4,
            },
        ),
        experiment(
            "seed_source_rows_rank10_recall_top6",
            "source_candidate_surfacing",
            "Use the highest full-first30 recall source row from the original source sweep.",
            {
                **clone(seed),
                "_kalcb.source.path": source_path,
                "_kalcb.source.section": "rows",
                "_kalcb.source.rank": 10,
            },
        ),
        experiment(
            "seed_avwap_r0p015",
            "route_family",
            "Swap the delayed branch to AVWAP reclaim at conservative risk.",
            replace_non_anchor_routes(seed, [avwap_015]),
        ),
        experiment(
            "seed_combo_pullback_avwap_r0p015_each",
            "route_family",
            "Test pullback + AVWAP complementarity at half-step route risk.",
            replace_non_anchor_routes(seed, [pullback_015, avwap_015]),
        ),
        experiment(
            "seed_avwap_r0p02_target60",
            "route_family_target",
            "AVWAP target-60 challenger mirroring the pullback target-60 result.",
            {**replace_non_anchor_routes(seed, [avwap_020_target60]), "kalcb.exit.target_r": 60.0},
        ),
        experiment(
            "seed_path_quality_off",
            "capture_exit",
            "Disable path-quality exit around the seed to verify whether it is protecting or cutting recoverable R.",
            {
                **clone(seed),
                "kalcb.exit.path_quality_enabled": False,
                "kalcb.exit.path_quality_min_hold_bars": 0,
                "kalcb.exit.path_quality_min_mfe_r": 0.0,
                "kalcb.exit.path_quality_min_giveback_r": 0.0,
                "kalcb.exit.path_quality.context_min": {},
                "kalcb.exit.path_quality_entry_route_modes": [],
            },
        ),
        experiment(
            "seed_partial12_25_be",
            "capture_exit",
            "Try a 12R quarter partial with breakeven protection to address EOD/giveback leakage.",
            {
                **clone(seed),
                "kalcb.exit.use_partial_takes": True,
                "kalcb.exit.partial_r_trigger": 12.0,
                "kalcb.exit.partial_fraction": 0.25,
                "kalcb.exit.partial_stop_to_breakeven": True,
                "kalcb.exit.partial_breakeven_buffer_r": 0.10,
            },
        ),
        experiment(
            "seed_positions6_sector4",
            "portfolio_constraint",
            "Tighten portfolio/sector breadth to test if concentration is degrading holdout path quality.",
            {
                **clone(seed),
                "kalcb.risk.max_positions": 6,
                "kalcb.risk.max_per_sector": 4,
            },
        ),
    ]
    return candidates


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {key: _num(metrics.get(key)) - _num(baseline.get(key)) for key in METRIC_KEYS}


def objective(row: dict[str, Any], base_train: dict[str, Any], base_holdout: dict[str, Any]) -> dict[str, Any]:
    train = row["train"]
    holdout = row["holdout"]
    train_delta = deltas(train, base_train)
    holdout_delta = deltas(holdout, base_holdout)
    train_trade_lift = train_delta.get("trade_count", 0.0) / max(_num(base_train.get("trade_count")), 1.0)
    holdout_trade_lift = holdout_delta.get("trade_count", 0.0) / max(_num(base_holdout.get("trade_count")), 1.0)
    train_dd_excess = max(_num(train.get("broker_max_drawdown_pct")) - 0.08, 0.0)
    holdout_dd_excess = max(_num(holdout.get("broker_max_drawdown_pct")) - 0.08, 0.0)
    holdout_floor_breach = max((_num(base_holdout.get("broker_net_return_pct")) - 0.003) - _num(holdout.get("broker_net_return_pct")), 0.0)
    score = 100.0 * (
        0.34 * holdout_delta.get("broker_net_return_pct", 0.0)
        + 0.24 * train_delta.get("broker_net_return_pct", 0.0)
        + 0.14 * min(train_trade_lift, 0.75)
        + 0.08 * min(holdout_trade_lift, 0.75)
        + 0.10 * train_delta.get("avg_mfe_capture", 0.0)
        + 0.10 * holdout_delta.get("avg_mfe_capture", 0.0)
        - 8.0 * train_dd_excess
        - 10.0 * holdout_dd_excess
        - 6.0 * holdout_floor_breach
    )
    gates = {
        "no_same_bar_fills": _num(train.get("same_bar_fill_count")) == 0.0 and _num(holdout.get("same_bar_fill_count")) == 0.0,
        "no_end_open_positions": _num(train.get("end_open_position_count")) == 0.0
        and _num(holdout.get("end_open_position_count")) == 0.0,
        "train_dd_le_8pct": _num(train.get("broker_max_drawdown_pct")) <= 0.08,
        "holdout_dd_le_8pct": _num(holdout.get("broker_max_drawdown_pct")) <= 0.08,
        "holdout_not_worse_than_30bps": _num(holdout.get("broker_net_return_pct"))
        >= _num(base_holdout.get("broker_net_return_pct")) - 0.003,
        "train_trades_ge_105": _num(train.get("trade_count")) >= 105.0,
        "holdout_trades_ge_baseline": _num(holdout.get("trade_count")) >= _num(base_holdout.get("trade_count")),
    }
    return {
        "score": score,
        "research_survivor": all(gates.values()),
        "gates": gates,
        "train_delta": train_delta,
        "holdout_delta": holdout_delta,
    }


def detail_digest(evaluator: RecoveryEvaluator, mutations: dict[str, Any], *, window: str) -> dict[str, Any]:
    plugin = evaluator.plugin if window == "train" else evaluator.plugin._validation_plugin()
    detail = plugin._evaluation_details.get(_mutation_key(mutations))
    if detail is None:
        return {}
    digest = dict(detail.replay_digest or {})
    return {
        "trade_count": digest.get("trade_count"),
        "entry_rejection_count": digest.get("entry_rejection_count"),
        "top_entry_rejection_reasons": digest.get("top_entry_rejection_reasons"),
        "top_entry_failed_gates": digest.get("top_entry_failed_gates"),
        "fill_count": digest.get("fill_count"),
        "same_bar_fill_count": digest.get("same_bar_fill_count"),
        "trade_hash": digest.get("trade_hash"),
    }


def progress(event: str, **extra: Any) -> None:
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **extra}
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    append_jsonl(OUT_DIR / "progress.jsonl", payload)


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    rows = sorted(payload["rows"], key=lambda row: _num(row.get("score")), reverse=True)
    survivors = [row for row in rows if row.get("research_survivor")]
    best = survivors[0] if survivors else (rows[0] if rows else {})
    q = payload.get("feature_quantiles") or {}
    pipeline = payload.get("positive_r_pipeline") or {}
    source = payload.get("source_row_digest") or {}
    lines = [
        "# KALCB Optimal Next Steps",
        "",
        "Research-only quantitative follow-up to the positive-R pipeline audit.",
        "",
        "## Decision",
        "",
        f"- Best ranked variant: `{best.get('label', '')}` (`{best.get('family', '')}`), survivor={best.get('research_survivor')}",
        f"- Train net/trades/capture: {_num(best.get('train', {}).get('broker_net_return_pct')):.2%} / {_num(best.get('train', {}).get('trade_count')):.0f} / {_num(best.get('train', {}).get('avg_mfe_capture')):.2%}",
        f"- Holdout net/trades/capture: {_num(best.get('holdout', {}).get('broker_net_return_pct')):.2%} / {_num(best.get('holdout', {}).get('trade_count')):.0f} / {_num(best.get('holdout', {}).get('avg_mfe_capture')):.2%}",
        "",
        "## Baseline",
        "",
        f"- Round-5 train: net={_num(baseline.get('train', {}).get('broker_net_return_pct')):.2%}, trades={_num(baseline.get('train', {}).get('trade_count')):.0f}, capture={_num(baseline.get('train', {}).get('avg_mfe_capture')):.2%}, DD={_num(baseline.get('train', {}).get('broker_max_drawdown_pct')):.2%}",
        f"- Round-5 holdout: net={_num(baseline.get('holdout', {}).get('broker_net_return_pct')):.2%}, trades={_num(baseline.get('holdout', {}).get('trade_count')):.0f}, capture={_num(baseline.get('holdout', {}).get('avg_mfe_capture')):.2%}, DD={_num(baseline.get('holdout', {}).get('broker_max_drawdown_pct')):.2%}",
        "",
        "## Ranked Variants",
        "",
        "| rank | label | family | survivor | train net | holdout net | train trades | holdout trades | train cap | holdout cap | score |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{row.get('label')}`",
                    str(row.get("family") or ""),
                    "Y" if row.get("research_survivor") else "N",
                    f"{_num(row.get('train', {}).get('broker_net_return_pct')):.2%}",
                    f"{_num(row.get('holdout', {}).get('broker_net_return_pct')):.2%}",
                    f"{_num(row.get('train', {}).get('trade_count')):.0f}",
                    f"{_num(row.get('holdout', {}).get('trade_count')):.0f}",
                    f"{_num(row.get('train', {}).get('avg_mfe_capture')):.2%}",
                    f"{_num(row.get('holdout', {}).get('avg_mfe_capture')):.2%}",
                    f"{_num(row.get('score')):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Feature Quantiles Used",
            "",
            f"- first30_rel_volume q80/q85/q90: {_num(q.get('first30_rel_volume', {}).get('q80')):.2f} / {_num(q.get('first30_rel_volume', {}).get('q85')):.2f} / {_num(q.get('first30_rel_volume', {}).get('q90')):.2f}",
            f"- first30_ret q80/q85/q90: {_num(q.get('first30_ret', {}).get('q80')):.2%} / {_num(q.get('first30_ret', {}).get('q85')):.2%} / {_num(q.get('first30_ret', {}).get('q90')):.2%}",
            f"- first30_signal_bar_cpr q75/q85/q90: {_num(q.get('first30_signal_bar_cpr', {}).get('q75')):.2f} / {_num(q.get('first30_signal_bar_cpr', {}).get('q85')):.2f} / {_num(q.get('first30_signal_bar_cpr', {}).get('q90')):.2f}",
            "",
            "## Positive-R Audit Inputs",
            "",
            f"- Pipeline report: `{pipeline.get('report_path', '')}`",
            f"- Largest stage leaks: `{pipeline.get('short_verdict', '')}`",
            "",
            "## Source-Row Recall Probe",
            "",
        ]
    )
    for row in list(source.get("top_net_with_recall_ge_25pct") or [])[:5]:
        lines.append(
            f"- rows[{row.get('rank')}]: recall={_num(row.get('full_first30_candidate_recall')):.2%}, "
            f"proxy_net={_num(row.get('portfolio_proxy_net_return_pct')):.2%}, "
            f"first30_min_ret={_num(row.get('first30_min_ret')):.2%}, "
            f"first30_min_gap={_num(row.get('first30_min_gap')):.2%}, top_n={row.get('first30_top_n')}"
        )
    lines.extend(
        [
            "",
            "## Next Action Contract",
            "",
            "- Promote only a survivor with no same-bar fills, no end-open positions, train/holdout DD <= 8%, and holdout no worse than round-5 by 30 bps.",
            "- If no breadth/source probe beats the seed on holdout, keep the conservative pullback seed and use source-row/structural-campaign work as research-only.",
            "- Do not increase risk on delayed route branches until the route converts better than the seed on holdout and MFE capture does not deteriorate.",
            "",
        ]
    )
    return "\n".join(lines)


def run(max_evals: int | None) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quantiles = feature_quantiles()
    base = _load_round_mutations(5)
    seed = load_seed_mutations(SEED_PATH)
    challenger = load_seed_mutations(CHALLENGER_PATH)
    candidates = build_candidates(base, seed, challenger, quantiles)
    if max_evals is not None:
        candidates = candidates[:max_evals]

    source_path = Path(str(seed.get("_kalcb.source.path") or base.get("_kalcb.source.path") or ""))
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    source_digest = source_row_digest(source_path)
    pipeline_report = PIPELINE_DIR / "kalcb_positive_r_pipeline_breakdown.md"
    pipeline_payload = {
        "report_path": str(pipeline_report),
        "short_verdict": "dataset->candidate flow/divergence/ADV, candidate->selected first30 return/gap/VWAP, selected->entered min_bar_ret/quality votes, entered->captured EOD/path giveback",
    }

    evaluator = RecoveryEvaluator(CONFIG_PATH, OUT_DIR, _round_source_ref(5), max_workers=1)
    progress("baseline_start")
    base_train, base_holdout = evaluator.evaluate_pair(base)
    progress(
        "baseline_done",
        train_net=base_train.get("broker_net_return_pct"),
        holdout_net=base_holdout.get("broker_net_return_pct"),
        train_trades=base_train.get("trade_count"),
        holdout_trades=base_holdout.get("trade_count"),
    )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        label = item["label"]
        mutations = item["mutations"]
        progress("candidate_start", index=index, total=len(candidates), label=label, family=item["family"])
        started = time.monotonic()
        train, holdout = evaluator.evaluate_pair(mutations)
        row = {
            "label": label,
            "family": item["family"],
            "purpose": item["purpose"],
            "mutation_hash": _mutation_key(mutations),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "train": compact_metrics(train),
            "holdout": compact_metrics(holdout),
            "train_replay_digest": detail_digest(evaluator, mutations, window="train"),
            "holdout_replay_digest": detail_digest(evaluator, mutations, window="holdout"),
        }
        row.update(objective(row, compact_metrics(base_train), compact_metrics(base_holdout)))
        rows.append(row)
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strategy": "kalcb",
            "round": 5,
            "profile": "optimal_next_steps_positive_r_followup",
            "feature_quantiles": quantiles,
            "source_row_digest": source_digest,
            "positive_r_pipeline": pipeline_payload,
            "baseline": {"train": compact_metrics(base_train), "holdout": compact_metrics(base_holdout)},
            "rows": rows,
        }
        write_json(OUT_DIR / "kalcb_optimal_next_steps_results.json", payload)
        (OUT_DIR / "kalcb_optimal_next_steps_report.md").write_text(render_markdown(payload), encoding="utf-8")
        progress(
            "candidate_done",
            label=label,
            score=row.get("score"),
            survivor=row.get("research_survivor"),
            train_net=train.get("broker_net_return_pct"),
            holdout_net=holdout.get("broker_net_return_pct"),
            train_trades=train.get("trade_count"),
            holdout_trades=holdout.get("trade_count"),
        )

    final_payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy": "kalcb",
        "round": 5,
        "profile": "optimal_next_steps_positive_r_followup",
        "feature_quantiles": quantiles,
        "source_row_digest": source_digest,
        "positive_r_pipeline": pipeline_payload,
        "baseline": {"train": compact_metrics(base_train), "holdout": compact_metrics(base_holdout)},
        "rows": sorted(rows, key=lambda row: _num(row.get("score")), reverse=True),
    }
    write_json(OUT_DIR / "kalcb_optimal_next_steps_results.json", final_payload)
    (OUT_DIR / "kalcb_optimal_next_steps_report.md").write_text(render_markdown(final_payload), encoding="utf-8")
    progress("complete", result_json=str(OUT_DIR / "kalcb_optimal_next_steps_results.json"))
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=None)
    args = parser.parse_args()
    run(args.max_evals)


if __name__ == "__main__":
    main()
