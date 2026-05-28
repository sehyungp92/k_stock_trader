from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.auto.shared.cache_keys import stable_signature

from kalcb_round3_oos_ablation import (
    CONFIG_PATH,
    OUT_DIR,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    ReplayHarness,
    _build_candidate_set,
    _clean_metric_row,
    _combined_score,
    _merge,
    _oos_repair_score,
    _read_json,
    _round_artifacts,
    _source_from_diag,
    _write_json,
    _window_config,
    normalize_runtime_config,
    load_yaml_config,
)


def _candidate(label: str, kind: str, mutations: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"label": label, "kind": kind, "mutations": dict(mutations), "reason": reason}


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = stable_signature(row["mutations"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _build_targeted_candidates(artifacts: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, metadata = _build_candidate_set(artifacts, quick=True)
    r2 = dict(artifacts["round_2"]["optimized"]["mutations"])
    r3 = dict(artifacts["round_3"]["optimized"]["mutations"])
    source_r2 = _source_from_diag(artifacts["round_2"]["diagnostics"])
    source_r3 = _source_from_diag(artifacts["round_3"]["diagnostics"])

    source_rank4 = source_r2
    source_rank0 = source_r3
    source_path = source_r3[SOURCE_PATH_MUTATION]
    source_rank5 = {SOURCE_PATH_MUTATION: source_path, SOURCE_SECTION_MUTATION: "top_portfolio_proxy", SOURCE_RANK_MUTATION: 5}

    bases = [
        ("final_rank0", _merge(r3, source_rank0)),
        ("final_rank4", _merge(r3, source_rank4)),
        ("round2_rank0", _merge(r2, source_rank0)),
        ("round2_rank4", _merge(r2, source_rank4)),
        ("round2_rank5", _merge(r2, source_rank5)),
    ]
    candidates: list[dict[str, Any]] = []

    cpr_values = (0.50, 0.55, 0.65, 0.75)
    min_ret_values = (0.005, 0.0100)
    target_values = (0.0, 20.0, 36.0, 70.0)
    cap_values = (0.30, 0.35, 0.50)
    ff_variants = {
        "ff_current": {
            "kalcb.exit.failed_followthrough_bars": 10,
            "kalcb.exit.failed_followthrough_mfe_r": 1.25,
            "kalcb.exit.failed_followthrough_close_r": -0.25,
        },
        "ff_fast": {
            "kalcb.exit.failed_followthrough_bars": 6,
            "kalcb.exit.failed_followthrough_mfe_r": 0.75,
            "kalcb.exit.failed_followthrough_close_r": -0.25,
        },
        "ff_round2": {
            "kalcb.exit.failed_followthrough_bars": 8,
            "kalcb.exit.failed_followthrough_mfe_r": 1.0,
            "kalcb.exit.failed_followthrough_close_r": -0.5,
        },
    }

    for base_name, base in bases:
        for cpr in cpr_values:
            for min_ret in min_ret_values:
                for target in target_values:
                    for cap in cap_values:
                        candidates.append(
                            _candidate(
                                f"{base_name}_cpr{cpr:g}_ret{min_ret:g}_target{target:g}_cap{cap:g}",
                                "targeted_quality_exit_risk",
                                _merge(
                                    base,
                                    {
                                        "kalcb.entry.min_first30_signal_cpr": cpr,
                                        "kalcb.entry.min_bar_ret": min_ret,
                                        "kalcb.exit.target_r": target,
                                        "kalcb.risk.max_position_notional_pct": cap,
                                    },
                                ),
                                "joint CPR/min-return/target/cap sweep after OOS weakness review",
                            )
                        )

    hard_stop_variants = {
        "first30_low": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "first30_low", "kalcb.exit.stop_pct": 0.003},
        "fixed003": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.003},
        "fixed005": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.005},
        "fixed007": {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.007},
    }
    for base_name, base in bases[:3]:
        for stop_name, stop_mut in hard_stop_variants.items():
            for cpr in (0.0, 0.50, 0.55, 0.65):
                for min_ret in (0.005, 0.010):
                    for target in (0.0, 20.0, 36.0):
                        candidates.append(
                            _candidate(
                                f"{base_name}_{stop_name}_cpr{cpr:g}_ret{min_ret:g}_target{target:g}",
                                "targeted_loss_containment",
                                _merge(
                                    base,
                                    stop_mut,
                                    {
                                        "kalcb.entry.min_first30_signal_cpr": cpr,
                                        "kalcb.entry.min_bar_ret": min_ret,
                                        "kalcb.exit.target_r": target,
                                    },
                                ),
                                "loss-containment combinations after failed-followthrough/eod loss cluster review",
                            )
                        )

    for base_name, base in bases:
        for ff_name, ff_mut in ff_variants.items():
            for cpr in (0.0, 0.55, 0.65, 0.75):
                for target in (0.0, 20.0, 36.0, 70.0):
                    candidates.append(
                        _candidate(
                            f"{base_name}_{ff_name}_cpr{cpr:g}_target{target:g}",
                            "targeted_failed_followthrough",
                            _merge(base, ff_mut, {"kalcb.entry.min_first30_signal_cpr": cpr, "kalcb.exit.target_r": target}),
                            "target-aware failed-followthrough crossed with CPR and target",
                        )
                    )

    metadata["targeted_phase_counts"] = dict(Counter(row["kind"] for row in candidates))
    return _dedupe(candidates), metadata


def _compact_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "kind": row.get("kind", ""),
        "reason": row.get("reason", ""),
        "metrics": _clean_metric_row(row["metrics"]),
        "source": row["source"],
        "mutations": row["mutations"],
        "score": row.get("oos_repair_score", 0.0),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Round 3 Targeted Repair Phase",
        "",
        f"- OOS candidates evaluated: {payload['counts']['oos_evaluated']}",
        f"- Train-confirmed candidates: {payload['counts']['train_confirmed']}",
        "",
        "## Top Confirmed",
        "| Label | OOS Net % | OOS Trades | OOS Win % | Train Net % | Train Trades | Train DD % |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["confirmed_ranked"][:30]:
        oos = row["oos"]["metrics"]
        train = row["train"]["metrics"]
        lines.append(
            f"| {row['label']} | {100.0 * float(oos.get('broker_net_return_pct', 0.0) or 0.0):.2f} | "
            f"{float(oos.get('trade_count', 0.0) or 0.0):.0f} | "
            f"{100.0 * float(oos.get('net_win_share', 0.0) or 0.0):.1f} | "
            f"{100.0 * float(train.get('broker_net_return_pct', 0.0) or 0.0):.2f} | "
            f"{float(train.get('trade_count', 0.0) or 0.0):.0f} | "
            f"{100.0 * float(train.get('broker_max_drawdown_pct', 0.0) or 0.0):.2f} |"
        )
    lines.extend(["", "## Top OOS By Net", "| Label | OOS Net % | Trades | Win % | DD % |", "| --- | --- | --- | --- | --- |"])
    for row in payload["oos_ranked_by_net"][:30]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['label']} | {100.0 * float(metrics.get('broker_net_return_pct', 0.0) or 0.0):.2f} | "
            f"{float(metrics.get('trade_count', 0.0) or 0.0):.0f} | "
            f"{100.0 * float(metrics.get('net_win_share', 0.0) or 0.0):.1f} | "
            f"{100.0 * float(metrics.get('broker_max_drawdown_pct', 0.0) or 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-train", type=int, default=120)
    parser.add_argument("--max-oos", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = normalize_runtime_config("kalcb", load_yaml_config(CONFIG_PATH))
    artifacts = _round_artifacts()
    candidates, metadata = _build_targeted_candidates(artifacts)
    if args.max_oos > 0:
        candidates = candidates[: args.max_oos]

    harness = ReplayHarness(base_config, OUT_DIR)
    harness.progress_path = OUT_DIR / "targeted_phase_progress.jsonl"
    if harness.progress_path.exists():
        harness.progress_path.unlink()
    default_source = metadata["source_r3"]
    harness._status("targeted_candidate_plan", total=len(candidates), counts=dict(Counter(row["kind"] for row in candidates)))

    oos_results: list[dict[str, Any]] = []
    results_by_label: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates, start=1):
        harness._status("targeted_oos_start", index=index, total=len(candidates), label=candidate["label"], kind=candidate["kind"])
        try:
            result = harness.evaluate(candidate["label"], candidate["mutations"], "oos", default_source)
            result["kind"] = candidate["kind"]
            result["reason"] = candidate["reason"]
            result["oos_repair_score"] = _oos_repair_score(result["metrics"])
            oos_results.append(result)
            results_by_label.setdefault(candidate["label"], {})["oos"] = result
            harness._status(
                "targeted_oos_done",
                index=index,
                label=candidate["label"],
                net=result["metrics"].get("broker_net_return_pct"),
                trades=result["metrics"].get("trade_count"),
                win=result["metrics"].get("net_win_share"),
            )
        except Exception as exc:
            harness._status("targeted_oos_error", index=index, label=candidate["label"], error=repr(exc))

    by_score = sorted(oos_results, key=lambda row: row["oos_repair_score"], reverse=True)
    by_net = sorted(oos_results, key=lambda row: row["metrics"].get("broker_net_return_pct", 0.0), reverse=True)
    by_balanced = [
        row
        for row in by_net
        if float(row["metrics"].get("trade_count", 0.0) or 0.0) >= 16.0
    ]
    train_labels = {row["label"] for row in by_score[: args.top_train // 2]}
    train_labels.update(row["label"] for row in by_net[: args.top_train // 3])
    train_labels.update(row["label"] for row in by_balanced[: args.top_train // 2])
    label_to_candidate = {row["label"]: row for row in candidates}
    train_candidates = [label_to_candidate[label] for label in train_labels if label in label_to_candidate]
    harness._status("targeted_train_plan", total=len(train_candidates))

    train_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(train_candidates, start=1):
        harness._status("targeted_train_start", index=index, total=len(train_candidates), label=candidate["label"], kind=candidate["kind"])
        try:
            result = harness.evaluate(candidate["label"], candidate["mutations"], "train", default_source)
            result["kind"] = candidate["kind"]
            result["reason"] = candidate["reason"]
            train_results.append(result)
            results_by_label.setdefault(candidate["label"], {})["train"] = result
            harness._status(
                "targeted_train_done",
                index=index,
                label=candidate["label"],
                net=result["metrics"].get("broker_net_return_pct"),
                trades=result["metrics"].get("trade_count"),
                win=result["metrics"].get("net_win_share"),
            )
        except Exception as exc:
            harness._status("targeted_train_error", index=index, label=candidate["label"], error=repr(exc))

    final_train = _read_json(OUT_DIR / "round3_oos_ablation_results.json")["results_by_label"]["round3_final"]["train"]["metrics"]
    confirmed = []
    for label, windows in results_by_label.items():
        if "oos" not in windows or "train" not in windows:
            continue
        confirmed.append(
            {
                "label": label,
                "kind": windows["oos"].get("kind", ""),
                "reason": windows["oos"].get("reason", ""),
                "combined_score": _combined_score(windows["oos"]["metrics"], windows["train"]["metrics"], final_train),
                "oos": {"metrics": _clean_metric_row(windows["oos"]["metrics"]), "source": windows["oos"]["source"], "mutations": windows["oos"]["mutations"]},
                "train": {"metrics": _clean_metric_row(windows["train"]["metrics"]), "source": windows["train"]["source"], "mutations": windows["train"]["mutations"]},
            }
        )
    confirmed.sort(key=lambda row: row["combined_score"], reverse=True)

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metadata": metadata,
        "counts": {"candidate_count": len(candidates), "oos_evaluated": len(oos_results), "train_confirmed": len(train_results)},
        "oos_ranked_by_score": [_compact_result(row) for row in by_score],
        "oos_ranked_by_net": [_compact_result(row) for row in by_net],
        "confirmed_ranked": confirmed,
        "top_recommendation": confirmed[0] if confirmed else {},
    }
    _write_json(OUT_DIR / "targeted_phase_results.json", payload)
    _write_json(OUT_DIR / "targeted_phase_recommended_mutations.json", payload["top_recommendation"])
    (OUT_DIR / "targeted_phase_summary.md").write_text(_markdown(payload), encoding="utf-8")
    harness._status("targeted_complete", result_path=str(OUT_DIR / "targeted_phase_results.json"))


if __name__ == "__main__":
    main()
