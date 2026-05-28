from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter
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
    score_fixed,
)


ROOT = Path(".")
ROUND5_DIR = ROOT / "data/backtests/output/kalcb/round_5"
OUT_DIR = ROUND5_DIR / "risk_discrimination_sweep"
OUT_JSON = OUT_DIR / "risk_discrimination_sweep_results.json"
OUT_MD = OUT_DIR / "risk_discrimination_sweep_report.md"

METRIC_KEYS = (
    "broker_net_return_pct",
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
    "same_bar_fill_count",
    "end_open_position_count",
    "immutable_score",
)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config/optimization/kalcb.yaml").read_text(encoding="utf-8")) or {}
    cfg["workers"] = 2
    cfg["validation_gate_enabled"] = False
    cfg["skip_initial_baseline_eval"] = True
    round5 = _read_json(ROUND5_DIR / "optimized_config.json")
    source = _source(round5["mutations"])

    train_cfg = deepcopy(cfg)
    train_cfg["fixed_candidate_source"] = source
    holdout_cfg = deepcopy(cfg)
    holdout = dict(holdout_cfg.get("baseline") or {})
    holdout_cfg["start"] = str(holdout["holdout_start"])
    holdout_cfg["end"] = str(holdout["holdout_end"])
    holdout_cfg["use_full_available_window"] = True
    holdout_cfg["validation_gate_enabled"] = False
    holdout_cfg["fixed_candidate_source"] = source

    _progress(started, "initialising")
    train = KALCBFixedTradePlanOptimizationPlugin(train_cfg, output_dir=OUT_DIR / "train_replay", max_workers=2)
    holdout_plugin = KALCBFixedTradePlanOptimizationPlugin(holdout_cfg, output_dir=OUT_DIR / "holdout_replay", max_workers=2)
    base = dict(round5["mutations"])
    base_train, base_train_rows = _eval(train, base)
    base_holdout, base_holdout_rows = _eval(holdout_plugin, base)

    rows = []
    experiments = _experiments(base)
    for idx, exp in enumerate(experiments, start=1):
        _progress(started, f"evaluating_{idx:02d}_{exp['name']}", index=idx, total=len(experiments))
        mutations = dict(base)
        mutations.update(exp["mutations"])
        rows.append(_eval_row(train, holdout_plugin, exp, mutations, base_train, base_holdout))
        _write(started, source, base_train, base_holdout, base_train_rows, base_holdout_rows, rows)
    payload = _write(started, source, base_train, base_holdout, base_train_rows, base_holdout_rows, rows)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    _progress(started, "done", index=len(experiments), total=len(experiments))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


def _experiments(base: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = base[SOURCE_PATH_MUTATION]
    return [
        {"name": "hard_stop_fixed_003", "family": "risk_stop", "mutations": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.003}},
        {"name": "hard_stop_fixed_006", "family": "risk_stop", "mutations": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.006}},
        {"name": "hard_stop_vwap", "family": "risk_stop", "mutations": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "vwap"}},
        {"name": "hard_stop_first30_low", "family": "risk_stop", "mutations": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "first30_low"}},
        {"name": "hard_stop_entry_low", "family": "risk_stop", "mutations": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "entry_low"}},
        {"name": "risk_fixed_006_no_hard_stop", "family": "risk_basis", "mutations": {"kalcb.exit.hard_stop_enabled": False, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.006}},
        {"name": "risk_first30_low_no_hard_stop", "family": "risk_basis", "mutations": {"kalcb.exit.hard_stop_enabled": False, "kalcb.exit.stop_mode": "first30_low"}},
        {"name": "quality_votes_7", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_quality_votes": 7}},
        {"name": "min_relvol_4", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_first30_rel_volume": 4.0}},
        {"name": "min_relvol_6", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_first30_rel_volume": 6.0}},
        {"name": "min_first30_ret_015", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_bar_ret": 0.015}},
        {"name": "min_cpr_085", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_first30_signal_cpr": 0.85}},
        {"name": "min_low_vs_prev_0", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_first30_low_vs_prev_close": 0.0}},
        {"name": "min_flow_0", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_flow_score": 0.0}},
        {"name": "min_accum_010", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_accumulation_score": 0.10}},
        {"name": "range_atr_cap_150", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.max_first30_range_atr": 1.50}},
        {"name": "combo_quality7_relvol4_cpr80", "family": "pre_entry_discrimination", "mutations": {"kalcb.entry.min_quality_votes": 7, "kalcb.entry.min_first30_rel_volume": 4.0, "kalcb.entry.min_first30_signal_cpr": 0.80}},
        {"name": "source_top_portfolio_proxy_rank1", "family": "source_row", "mutations": {SOURCE_PATH_MUTATION: source_path, SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 1}},
        {"name": "source_top_slot_return_rank0", "family": "source_row", "mutations": {SOURCE_PATH_MUTATION: source_path, SOURCE_SECTION_MUTATION: "top_slot_return", SOURCE_RANK_MUTATION: 0}},
        {"name": "source_top_mfe_rank4", "family": "source_row", "mutations": {SOURCE_PATH_MUTATION: source_path, SOURCE_SECTION_MUTATION: "top_mfe", SOURCE_RANK_MUTATION: 4}},
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(mutations: dict[str, Any]) -> dict[str, Any]:
    return {"path": mutations[SOURCE_PATH_MUTATION], "section": mutations[SOURCE_SECTION_MUTATION], "rank": int(mutations[SOURCE_RANK_MUTATION])}


def _eval(plugin: KALCBFixedTradePlanOptimizationPlugin, mutations: dict[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    metrics = plugin.evaluate_mutations(mutations)
    detail = plugin._evaluation_details[_mutation_key(mutations)]
    return dict(metrics), tuple(detail.trade_rows)


def _eval_row(train_plugin, holdout_plugin, exp, mutations, base_train, base_holdout) -> dict[str, Any]:
    try:
        train, train_rows = _eval(train_plugin, mutations)
        holdout, holdout_rows = _eval(holdout_plugin, mutations)
        td = _delta(_compact(train), _compact(base_train))
        hd = _delta(_compact(holdout), _compact(base_holdout))
        return {
            "name": exp["name"],
            "family": exp["family"],
            "mutation_hash": stable_signature(exp["mutations"]),
            "mutations": exp["mutations"],
            "train": _compact(train),
            "holdout": _compact(holdout),
            "train_delta": td,
            "holdout_delta": hd,
            "train_path": _path(train_rows),
            "holdout_path": _path(holdout_rows),
            "score": _score(td, hd),
            "promotable_research_candidate": _promotable(td, hd),
        }
    except Exception as exc:
        return {"name": exp["name"], "family": exp["family"], "mutations": exp["mutations"], "error": f"{type(exc).__name__}: {exc}", "score": -999.0}


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
    out["immutable_score"] = metrics.get("immutable_score", score_fixed(metrics))
    return out


def _delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) - float(baseline[key]) for key, value in metrics.items() if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))}


def _score(td: dict[str, float], hd: dict[str, float]) -> float:
    dd = max(td.get("broker_max_drawdown_pct", 0.0), 0.0) + max(hd.get("broker_max_drawdown_pct", 0.0), 0.0)
    return 100.0 * (
        0.30 * math.tanh(td.get("broker_net_return_pct", 0.0) / 0.07)
        + 0.30 * math.tanh(hd.get("broker_net_return_pct", 0.0) / 0.035)
        + 0.14 * math.tanh(td.get("avg_trade_net_pct", 0.0) / 0.003)
        + 0.12 * math.tanh(hd.get("avg_trade_net_pct", 0.0) / 0.003)
        + 0.06 * math.tanh((td.get("avg_mfe_capture", 0.0) + hd.get("avg_mfe_capture", 0.0)) / 0.05)
        - 0.08 * math.tanh(dd / 0.012)
    )


def _promotable(td: dict[str, float], hd: dict[str, float]) -> bool:
    return (
        td.get("broker_net_return_pct", 0.0) > 0.0
        and hd.get("broker_net_return_pct", 0.0) > 0.0
        and td.get("avg_trade_net_pct", 0.0) >= -0.0005
        and hd.get("avg_trade_net_pct", 0.0) >= -0.0005
        and td.get("broker_max_drawdown_pct", 0.0) <= 0.006
        and hd.get("broker_max_drawdown_pct", 0.0) <= 0.006
        and td.get("worst_fold_net", 0.0) >= -0.004
    )


def _path(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    r = [_num(row.get("r")) for row in rows]
    losers = [row for row in rows if _num(row.get("r")) < 0]
    return {
        "trades": len(rows),
        "total_r": sum(r),
        "avg_r": _mean(r),
        "loser_share": len(losers) / max(len(rows), 1),
        "loser_avg_mae_r": _mean([_num(row.get("mae_r")) for row in losers]),
        "avg_mfe_r": _mean([_num(row.get("mfe_r")) for row in rows]),
        "avg_mae_r": _mean([_num(row.get("mae_r")) for row in rows]),
        "exit_counts": dict(sorted(Counter(str(row.get("exit_reason") or "unknown") for row in rows).items())),
    }


def _write(started, source, base_train, base_holdout, base_train_rows, base_holdout_rows, rows):
    valid = [row for row in rows if "error" not in row]
    best = sorted(valid, key=lambda row: row.get("score", -999.0), reverse=True)[:10]
    promotable = [row for row in valid if row.get("promotable_research_candidate")]
    payload = {
        "elapsed_seconds": round(time.time() - started, 3),
        "source_ref": source,
        "baseline": {"train": _compact(base_train), "holdout": _compact(base_holdout), "train_path": _path(base_train_rows), "holdout_path": _path(base_holdout_rows)},
        "rows": rows,
        "summary": {"tested": len(rows), "valid": len(valid), "promotable": [_summary(row) for row in promotable], "best": [_summary(row) for row in best]},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    td, hd = row.get("train_delta", {}), row.get("holdout_delta", {})
    return {
        "name": row.get("name"),
        "family": row.get("family"),
        "score": row.get("score"),
        "train_net_delta": td.get("broker_net_return_pct"),
        "holdout_net_delta": hd.get("broker_net_return_pct"),
        "train_dd_delta": td.get("broker_max_drawdown_pct"),
        "holdout_dd_delta": hd.get("broker_max_drawdown_pct"),
        "train_trade_delta": td.get("trade_count"),
        "holdout_trade_delta": hd.get("trade_count"),
        "promotable_research_candidate": row.get("promotable_research_candidate"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# KALCB risk/discrimination/source sweep", "", f"Tested {payload['summary']['tested']} hypotheses in {payload['elapsed_seconds']}s.", "", "## Best", "", "| Candidate | Family | Train net d | Holdout net d | Train DD d | Holdout DD d | Trade d T/H |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]["best"]:
        lines.append(f"| {row['name']} | {row['family']} | {_pct(row['train_net_delta'])} | {_pct(row['holdout_net_delta'])} | {_pct(row['train_dd_delta'])} | {_pct(row['holdout_dd_delta'])} | {_f(row['train_trade_delta'],0)}/{_f(row['holdout_trade_delta'],0)} |")
    lines.extend(["", "## All Rows", "", "| Candidate | Family | Train net d | Holdout net d | Train avg d | Holdout avg d | Train DD d | Holdout DD d |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["rows"]:
        if "error" in row:
            lines.append(f"| {row['name']} | {row['family']} | error | error | error | error | error | error |")
            continue
        td, hd = row["train_delta"], row["holdout_delta"]
        lines.append(f"| {row['name']} | {row['family']} | {_pct(td.get('broker_net_return_pct'))} | {_pct(hd.get('broker_net_return_pct'))} | {_pct(td.get('avg_trade_net_pct'))} | {_pct(hd.get('avg_trade_net_pct'))} | {_pct(td.get('broker_max_drawdown_pct'))} | {_pct(hd.get('broker_max_drawdown_pct'))} |")
    return "\n".join(lines) + "\n"


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(values)) if values else 0.0


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _f(value: Any, places: int = 2) -> str:
    return f"{_num(value):.{places}f}"


def _progress(started: float, stage: str, **extra: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "progress.json").write_text(json.dumps({"stage": stage, "elapsed_seconds": round(time.time() - started, 3), **extra}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
