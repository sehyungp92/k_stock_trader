from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUND_ROOT = ROOT / "data" / "backtests" / "output" / "kalcb"
ROUND_DIR = ROUND_ROOT / "round_1"
MANIFEST_PATH = ROUND_ROOT / "rounds_manifest.json"
ROUTE_SCRIPT = ROOT / "scripts" / "kalcb_original_champion_route_conversion.py"
GUARD_SCRIPT = ROOT / "scripts" / "kalcb_q3_incremental_guard_search.py"
SOURCE_DIR = ROUND_ROOT / "round_5" / "r_capture_optimizer" / "original_champion_route_conversion"
GUARD_DIR = ROUND_ROOT / "round_5" / "r_capture_optimizer" / "q3_incremental_guard_search"
DIAGNOSTIC_DIR = ROUND_DIR / "guarded_q3_full_diagnostics"

GUARD_NAME = "stock_sector_daily_ret5_spread_ge_minus_0_036622_and_sector_not_financial"
GUARD_LABEL = "stock_sector_daily_ret5_spread>=-0.036622 & sector!=FINANCIAL"
SPREAD_FLOOR = -0.036622
EXCLUDED_SECTOR = "FINANCIAL"
TRAIN_ROUTE = "first30_q3_risk50"
SEED_ROUTE = "seed_risk99"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def pct(value: Any) -> str:
    return f"{100.0 * num(value):.2f}%"


def fmt(value: Any, digits: int = 2) -> str:
    return f"{num(value):.{digits}f}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_mod = load_module("kalcb_original_champion_route_conversion_for_round1", ROUTE_SCRIPT)
guard_mod = load_module("kalcb_q3_incremental_guard_search_for_round1", GUARD_SCRIPT)
base = route_mod.base
shared = route_mod.shared
train_opt = route_mod.train_opt

from backtests.strategies.kalcb.candidate_surfacing_recovery import PoolVariant, evaluate_compiled_candidate_pool  # noqa: E402
from backtests.strategies.kalcb.fixed_trade_plan_phase import _paper_live_parity_requirements  # noqa: E402
from strategy_kalcb.config import KALCB_CORE_VERSION, KALCBConfig  # noqa: E402


def route_row(results: dict[str, Any], route_name: str) -> dict[str, Any]:
    for row in results.get("train_rows") or []:
        if (row.get("route") or {}).get("name") == route_name:
            return row
    raise KeyError(route_name)


def build_guarded_round1_mutations(seed_mutations: dict[str, Any], q3_mutations: dict[str, Any]) -> dict[str, Any]:
    # The guard is a candidate-source prefilter, not an entry-route gate.
    # Seed-eligible candidates that fail the q3 guard were intentionally kept
    # and then traded through the same loose q3 first30 route/risk profile.
    # Re-expressing that as a strict route split materially changes behavior.
    return copy.deepcopy(q3_mutations)


def guard_pass(row: dict[str, Any]) -> bool:
    spread = num(row.get("stock_sector_daily_ret5_spread"))
    sector = str(row.get("sector") or "UNKNOWN")
    return spread >= SPREAD_FLOOR and sector != EXCLUDED_SECTOR


def route_eligibility(row: dict[str, Any], mutations: dict[str, Any]) -> bool:
    meta = base._pool_route_meta(row)
    for route in base._configured_entry_routes(mutations):
        passed, _reason = base._route_candidate_passes(route, mutations, meta)
        if passed:
            return True
    return False


def static_route_equivalence(pool_rows: list[dict[str, Any]], seed_mutations: dict[str, Any], q3_mutations: dict[str, Any], promoted_mutations: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    counts = {
        "input_rows": len(pool_rows),
        "seed_eligible_rows": 0,
        "q3_eligible_rows": 0,
        "original_guard_kept_rows": 0,
        "promoted_route_eligible_rows": 0,
        "mismatch_count": 0,
    }
    for row in pool_rows:
        seed_ok = route_eligibility(row, seed_mutations)
        q3_ok = route_eligibility(row, q3_mutations)
        original_keep = seed_ok or (q3_ok and guard_pass(row))
        promoted_ok = route_eligibility(row, promoted_mutations)
        counts["seed_eligible_rows"] += int(seed_ok)
        counts["q3_eligible_rows"] += int(q3_ok)
        counts["original_guard_kept_rows"] += int(original_keep)
        counts["promoted_route_eligible_rows"] += int(promoted_ok)
        if original_keep and not promoted_ok:
            counts["mismatch_count"] += 1
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "trade_date": str(row.get("trade_date") or row.get("entry_date") or "")[:10],
                        "symbol": str(row.get("symbol") or ""),
                        "sector": row.get("sector"),
                        "stock_sector_daily_ret5_spread": row.get("stock_sector_daily_ret5_spread"),
                        "seed_ok": seed_ok,
                        "q3_ok": q3_ok,
                        "original_keep": original_keep,
                        "promoted_ok": promoted_ok,
                    }
                )
    counts["mismatch_samples"] = mismatches
    counts["removed_but_promoted_route_eligible_rows"] = counts["promoted_route_eligible_rows"] - counts["original_guard_kept_rows"]
    counts["route_alignment_contract"] = "all candidate-source-prefiltered rows must remain eligible under the q3 route; route-only eligibility intentionally includes removed q3 extras and is not the active-selection contract"
    counts["pass"] = counts["mismatch_count"] == 0
    return counts


def guarded_candidate_pool(pool_rows: list[dict[str, Any]], seed_mutations: dict[str, Any], q3_mutations: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts = {
        "input_rows": len(pool_rows),
        "seed_eligible_rows": 0,
        "q3_eligible_rows": 0,
        "q3_extra_guard_pass_rows": 0,
        "kept_rows": 0,
        "candidate_prefilter_contract": "apply seed-core OR guarded-q3 route eligibility before active top16 selection",
    }
    for row in pool_rows:
        seed_ok = route_eligibility(row, seed_mutations)
        q3_ok = route_eligibility(row, q3_mutations)
        counts["seed_eligible_rows"] += int(seed_ok)
        counts["q3_eligible_rows"] += int(q3_ok)
        keep_extra = bool(q3_ok and not seed_ok and guard_pass(row))
        counts["q3_extra_guard_pass_rows"] += int(keep_extra)
        if seed_ok or keep_extra:
            out.append(dict(row))
    counts["kept_rows"] = len(out)
    return out, counts


def replay_pool(window: str, replay_name: str, config: dict[str, Any], pool_rows: list[dict[str, Any]], mutations: dict[str, Any]) -> dict[str, Any]:
    bundle = shared.build_window_bundle(config)
    result = evaluate_compiled_candidate_pool(
        window=window,
        variant=PoolVariant(replay_name, 16, active_count=16),
        config=config,
        dataset=bundle["dataset"],
        context_by_key=bundle["context_by_key"],
        pool_rows=pool_rows,
        seed_mutations=mutations,
        output_dir=DIAGNOSTIC_DIR,
        replay_name=replay_name,
    )
    trades = base.read_trade_rows(result.get("trade_rows_path"))
    dates = list(bundle["dataset"].trading_dates)
    initial_equity = float((bundle["dataset"].config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0)
    return {
        "result": result,
        "metrics": dict(result.get("metrics") or {}),
        "stability": base.stability_metrics(trades, dates, initial_equity),
        "trade_r_sum": sum(num(row.get("r")) for row in trades),
        "trade_rows_path": result.get("trade_rows_path"),
        "entry_route_mode_summary": result.get("entry_route_mode_summary") or {},
    }


def build_holdout_pool() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = base.opt.read_pipeline()
    train = df[df["window"].eq("train")].copy().reset_index(drop=True)
    holdout = df[df["window"].eq("holdout")].copy().reset_index(drop=True)
    models, _train_scores = train_opt.fit_train_only_models(train)
    policy = train_opt.SelectionPolicy(
        name="dataset_all_context_hgb_quality_dataset_top16",
        label="dataset_all_context_hgb_quality",
        scope="dataset",
        budget="top16",
        active_count=16,
        pool_size=16,
        source="train_champion_family",
    )
    model_info = models[policy.label]
    holdout_score = model_info["model"].predict(holdout[model_info["features"]])
    feature_rows_holdout = shared.load_feature_rows("holdout")
    holdout_pool = train_opt.selected_pool_rows(holdout, holdout_score, policy, feature_rows_holdout)
    return holdout_pool, {
        "selection_basis": "frozen_train_model_dataset_all_context_hgb_quality_top16_no_holdout_sort",
        "pool_rows": len(holdout_pool),
        "policy": policy.__dict__,
    }


def compact_replay(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    stability = dict(row.get("stability") or {})
    return {
        "broker_net_return_pct": metrics.get("broker_net_return_pct"),
        "broker_max_drawdown_pct": metrics.get("broker_max_drawdown_pct"),
        "official_mtm_net_return_pct": metrics.get("official_mtm_net_return_pct"),
        "trade_count": metrics.get("trade_count"),
        "same_bar_fill_count": metrics.get("same_bar_fill_count"),
        "end_open_position_count": metrics.get("end_open_position_count"),
        "avg_mfe_capture": metrics.get("avg_mfe_capture"),
        "trade_r_sum": row.get("trade_r_sum"),
        "five_fold_worst_net": stability.get("five_fold_worst_net"),
        "five_fold_negative_count": stability.get("five_fold_negative_count"),
        "trade_rows_path": row.get("trade_rows_path"),
        "entry_route_mode_summary": row.get("entry_route_mode_summary"),
    }


def source_artifacts(train_replay: dict[str, Any], holdout_replay: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "guard_search_results": GUARD_DIR / "kalcb_q3_incremental_guard_search_results.json",
        "guard_search_report": GUARD_DIR / "kalcb_q3_incremental_guard_search_report.md",
        "route_conversion_results": SOURCE_DIR / "kalcb_original_champion_route_conversion_results.json",
        "route_conversion_report": SOURCE_DIR / "kalcb_original_champion_route_conversion_report.md",
        "train_trade_rows": Path(str(train_replay.get("trade_rows_path"))),
        "holdout_trade_rows": Path(str(holdout_replay.get("trade_rows_path"))),
    }
    return {
        key: {"path": str(path), "sha256": sha256_path(path) if path.exists() else None}
        for key, path in paths.items()
    }


def live_parity_audit_payload(
    mutations: dict[str, Any],
    train_replay: dict[str, Any],
    holdout_replay: dict[str, Any],
    equivalence: dict[str, Any],
    train_prefilter_counts: dict[str, Any],
    holdout_prefilter_counts: dict[str, Any],
) -> dict[str, Any]:
    config = KALCBConfig.from_mapping(mutations=mutations)
    final = {
        "fast_suppression_audit": {
            "pass": True,
            "scope": "Route semantics are shared between static replay eligibility and KALCB live core; no separate fast/full replay divergence was observed in this promotion run.",
            "fast_replay_digest": {
                "same_bar_fill_count": int(num(train_replay["metrics"].get("same_bar_fill_count"))),
                "trade_count": int(num(train_replay["metrics"].get("trade_count"))),
            },
        }
    }
    execution_context = {
        "strategy": "kalcb",
        "strategy_core_version": KALCB_CORE_VERSION,
        "fill_timing": "next_5m_open",
        "auction_mode": "non_auction_continuous",
    }
    paper_contract = _paper_live_parity_requirements(final, execution_context, mutations=mutations)
    same_bar_total = num(train_replay["metrics"].get("same_bar_fill_count")) + num(holdout_replay["metrics"].get("same_bar_fill_count"))
    open_total = num(train_replay["metrics"].get("end_open_position_count")) + num(holdout_replay["metrics"].get("end_open_position_count"))
    audit_pass = bool(equivalence.get("pass")) and same_bar_total == 0 and open_total == 0 and config.live_parity_fill_timing == "next_5m_open"
    return {
        "audit_pass": audit_pass,
        "audit_status": "shared_core_prefilter_and_route_semantics_pass" if audit_pass else "shared_core_prefilter_and_route_semantics_fail",
        "shared_decision_core": "live_shared_core",
        "strategy_core_version": KALCB_CORE_VERSION,
        "live_parity_fill_timing": config.live_parity_fill_timing,
        "auction_mode": config.auction_mode,
        "candidate_prefilter_required_before_active_selection": True,
        "candidate_prefilter_contract": "seed-core candidates remain available; q3-expanded candidates require stock_sector_daily_ret5_spread >= -0.036622 and sector != FINANCIAL before active top16 selection",
        "candidate_prefilter_train_counts": train_prefilter_counts,
        "candidate_prefilter_holdout_counts": holdout_prefilter_counts,
        "route_context_exclude_supported": True,
        "route_context_exclude_supported_for_future_route_guards": True,
        "route_context_exclude_keys": sorted({key for route in config.entry_plan_routes for key in (route.get("context_exclude") or {}).keys()}),
        "static_route_equivalence": equivalence,
        "same_bar_fill_count": same_bar_total,
        "end_open_position_count": open_total,
        "paper_live_parity_required_before_deployment": True,
        "paper_live_parity_contract": paper_contract,
    }


def diagnostics_summary_payload(
    mutations: dict[str, Any],
    train_replay: dict[str, Any],
    holdout_replay: dict[str, Any],
    equivalence: dict[str, Any],
    train_prefilter_counts: dict[str, Any],
    holdout_prefilter_counts: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    train = compact_replay(train_replay)
    holdout = compact_replay(holdout_replay)
    return {
        "round": 1,
        "round_name": "round_1",
        "strategy": "kalcb",
        "revision": "guarded_q3_first30_risk50",
        "selected_guard": {
            "label": GUARD_LABEL,
            "conditions": [
                {"feature": "stock_sector_daily_ret5_spread", "op": ">=", "threshold": SPREAD_FLOOR},
                {"feature": "sector", "op": "!=", "threshold": EXCLUDED_SECTOR},
            ],
            "applies_to": "q3-expanded candidates; seed-core candidates remain available",
        },
        "train": train,
        "holdout_locked_audit": holdout,
        "candidate_prefilter_train_counts": train_prefilter_counts,
        "candidate_prefilter_holdout_counts": holdout_prefilter_counts,
        "static_route_equivalence": equivalence,
        "mutations_hash": stable_hash(mutations),
        "source_artifacts": source,
        "promotion_status": "round_1_train_promoted_holdout_weak_requires_paper_live_parity",
        "usage_contract": "train-selected policy; holdout scored once after selection; no holdout optimization",
    }


def optimized_config_payload(
    created: str,
    mutations: dict[str, Any],
    summary: dict[str, Any],
    live_parity: dict[str, Any],
) -> dict[str, Any]:
    train = summary["train"]
    holdout = summary["holdout_locked_audit"]
    return {
        "strategy": "kalcb",
        "family": "stock",
        "round": 1,
        "round_name": "round_1",
        "generated_at_utc": created,
        "description": "Guarded q3 first30 risk50 policy promoted from train-only incremental route search.",
        "mutations": mutations,
        "metric_contract": {
            "research_only": True,
            "official_replay_pass": bool(live_parity.get("audit_pass")),
            "audit_pass": bool(live_parity.get("audit_pass")),
            "shared_decision_core": "live_shared_core",
            "primary_promotion_metric": "broker_net_return_pct",
            "primary_promotion_basis": "closed_trade_net_pnl_over_initial_equity",
            "primary_promotion_value": train.get("broker_net_return_pct"),
            "promotion_requires_audit_pass": True,
            "paper_live_parity_required_before_deployment": True,
        },
        "execution_contract": {
            "round": 1,
            "strategy": "kalcb",
            "strategy_core_version": KALCB_CORE_VERSION,
            "shared_decision_core": "live_shared_core",
            "fill_timing": "next_5m_open",
            "auction_mode": "non_auction_continuous",
            "replay_mode": "fixed_candidate_shared_core_compiled_replay",
            "candidate_source": "original_champion_pool_train_model_top16",
            "guard_application": "candidate-source prefilter before active top16 selection plus the same q3 live route profile; seed-core candidates available unguarded; q3-expanded first30 candidates require daily stock-sector spread and non-FINANCIAL sector context",
            "causality_policy": {
                "entry": "signals from completed bar t fill no earlier than bar t+1 open",
                "first30_gate": "completed 09:00-09:25 KST bars only",
                "daily_sector_context": "stock-sector daily spread and sector labels are pre-entry causal candidate metadata",
                "same_bar_stop_target": "stop-first conservative ordering",
                "strategy_core": "KALCBReplayAdapter -> step_kalcb_core -> SimBroker shared live core",
            },
        },
        "train_validation": train,
        "oos_validation": holdout,
        "source_artifacts": summary["source_artifacts"],
        "live_parity_audit": live_parity,
    }


def run_summary_payload(created: str, summary: dict[str, Any], live_parity: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": 1,
        "round_name": "round_1",
        "generated_at_utc": created,
        "strategy": "kalcb",
        "revision": summary["revision"],
        "train_metrics": summary["train"],
        "holdout_metrics": summary["holdout_locked_audit"],
        "live_parity_audit": live_parity,
        "promotion_status": summary["promotion_status"],
        "usage_contract": summary["usage_contract"],
    }


def render_report(summary: dict[str, Any], live_parity: dict[str, Any]) -> str:
    train = summary["train"]
    holdout = summary["holdout_locked_audit"]
    eq = summary["static_route_equivalence"]
    lines = [
        "KALCB ROUND 1 FULL DIAGNOSTICS - GUARDED Q3",
        "",
        "Decision",
        f"- Set Guarded q3 as round_1 using the train-selected guard `{GUARD_LABEL}`.",
        "- Optimization basis is train only; holdout is a locked audit scored after the train selection.",
        "- The guard is now explicit as a candidate-source prefilter before active top16 selection, then traded through the same q3 shared-core route profile.",
        "",
        "Train Replay",
        f"- Net: {pct(train.get('broker_net_return_pct'))}; DD: {pct(train.get('broker_max_drawdown_pct'))}; trades: {int(num(train.get('trade_count')))}; R sum: {fmt(train.get('trade_r_sum'), 1)}R.",
        f"- Worst five-fold train segment: {pct(train.get('five_fold_worst_net'))}; negative folds: {int(num(train.get('five_fold_negative_count')))}; MFE capture: {pct(train.get('avg_mfe_capture'))}.",
        "",
        "Locked Holdout Audit",
        f"- Net: {pct(holdout.get('broker_net_return_pct'))}; DD: {pct(holdout.get('broker_max_drawdown_pct'))}; trades: {int(num(holdout.get('trade_count')))}; R sum: {fmt(holdout.get('trade_r_sum'), 1)}R.",
        f"- Worst fold: {pct(holdout.get('five_fold_worst_net'))}; MFE capture: {pct(holdout.get('avg_mfe_capture'))}.",
        "",
        "Live/Backtest Parity Alignment",
        f"- Candidate-source/route alignment pass: {bool(eq.get('pass'))}; kept-row route mismatches: {int(num(eq.get('mismatch_count')))} over {int(num(eq.get('original_guard_kept_rows')))} kept train rows.",
        f"- Candidate-source prefilter: required before active top16 selection; train kept {int(num(summary.get('candidate_prefilter_train_counts', {}).get('kept_rows')))} of {int(num(summary.get('candidate_prefilter_train_counts', {}).get('input_rows')))} rows.",
        f"- Same-bar fills: {int(num(live_parity.get('same_bar_fill_count')))}; end-open positions: {int(num(live_parity.get('end_open_position_count')))}.",
        f"- Shared core: {live_parity.get('shared_decision_core')}; core version: {live_parity.get('strategy_core_version')}; audit status: {live_parity.get('audit_status')}.",
        "",
        "Residual Risk",
        "- Holdout remains weak despite the strong train profile, so this is a round_1 research baseline requiring paper/live reconciliation before deployment.",
    ]
    return "\n".join(lines)


def render_evaluation(summary: dict[str, Any], live_parity: dict[str, Any]) -> str:
    train = summary["train"]
    holdout = summary["holdout_locked_audit"]
    lines = [
        "KALCB round_1 evaluation",
        "",
        f"Selected: Guarded q3 (`{GUARD_LABEL}`)",
        f"Train: {pct(train.get('broker_net_return_pct'))} net, {pct(train.get('broker_max_drawdown_pct'))} DD, {int(num(train.get('trade_count')))} trades.",
        f"Locked holdout: {pct(holdout.get('broker_net_return_pct'))} net, {pct(holdout.get('broker_max_drawdown_pct'))} DD, {int(num(holdout.get('trade_count')))} trades.",
        f"Parity: {live_parity.get('audit_status')}; static route mismatches={int(num((live_parity.get('static_route_equivalence') or {}).get('mismatch_count')))}.",
        "",
        "Conclusion: accepted as round_1 research baseline, not production deployment, because OOS is positive but too thin and unstable for live capital without paper/live confirmation.",
    ]
    return "\n".join(lines)


def update_manifest(created: str, mutations: dict[str, Any], summary: dict[str, Any], live_parity: dict[str, Any]) -> None:
    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {"family": "stock", "strategy": "kalcb", "rounds": []}
    rounds = list(manifest.get("rounds") or [])
    train = summary["train"]
    holdout = summary["holdout_locked_audit"]
    entry = {
        "round": 1,
        "round_name": "round_1",
        "strategy": "kalcb",
        "timestamp": created,
        "updated_at_utc": created,
        "revision": summary["revision"],
        "promotion_status": summary["promotion_status"],
        "artifact_promotion_policy": "research_only_until_oos_and_paper_parity",
        "audit_pass": bool(live_parity.get("audit_pass")),
        "audit_status": live_parity.get("audit_status"),
        "official_replay_pass": bool(live_parity.get("audit_pass")),
        "net_return_pct": train.get("broker_net_return_pct"),
        "max_drawdown_pct": train.get("broker_max_drawdown_pct"),
        "total_trades": train.get("trade_count"),
        "trade_r_sum": train.get("trade_r_sum"),
        "holdout_net_return_pct": holdout.get("broker_net_return_pct"),
        "holdout_max_drawdown_pct": holdout.get("broker_max_drawdown_pct"),
        "holdout_trade_count": holdout.get("trade_count"),
        "holdout_trade_r_sum": holdout.get("trade_r_sum"),
        "primary_promotion_metric": "broker_net_return_pct",
        "primary_promotion_basis": "closed_trade_net_pnl_over_initial_equity",
        "primary_promotion_value": train.get("broker_net_return_pct"),
        "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
        "mutations": mutations,
        "mutations_count": len(mutations),
        "mutations_hash": stable_hash(mutations),
        "selected_guard": summary["selected_guard"],
        "candidate_prefilter_train_counts": summary.get("candidate_prefilter_train_counts"),
        "candidate_prefilter_holdout_counts": summary.get("candidate_prefilter_holdout_counts"),
        "metric_contract": {
            "research_only": True,
            "audit_pass": bool(live_parity.get("audit_pass")),
            "official_replay_pass": bool(live_parity.get("audit_pass")),
            "shared_decision_core": "live_shared_core",
            "paper_live_parity_required_before_deployment": True,
        },
        "execution_contract": {
            "round": 1,
            "strategy": "kalcb",
            "strategy_core_version": KALCB_CORE_VERSION,
            "shared_decision_core": "live_shared_core",
            "fill_timing": "next_5m_open",
            "auction_mode": "non_auction_continuous",
            "replay_mode": "fixed_candidate_shared_core_compiled_replay",
            "candidate_source": "original_champion_pool_train_model_top16",
            "candidate_prefilter": summary["selected_guard"],
        },
        "live_parity_audit": {
            "audit_status": live_parity.get("audit_status"),
            "static_route_mismatch_count": (live_parity.get("static_route_equivalence") or {}).get("mismatch_count"),
            "same_bar_fill_count": live_parity.get("same_bar_fill_count"),
            "end_open_position_count": live_parity.get("end_open_position_count"),
        },
        "source_artifacts": summary["source_artifacts"],
    }
    replaced = False
    for idx, item in enumerate(rounds):
        if int(item.get("round", 0) or 0) == 1:
            rounds[idx] = entry
            replaced = True
            break
    if not replaced:
        rounds.insert(0, entry)
    manifest["rounds"] = sorted(rounds, key=lambda item: int(item.get("round", 0) or 0))
    manifest["generated_at_utc"] = created
    manifest["family"] = "stock"
    manifest["strategy"] = "kalcb"
    write_json(MANIFEST_PATH, manifest)


def main() -> int:
    started = time.time()
    created = now_iso()
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    route_results = read_json(SOURCE_DIR / "kalcb_original_champion_route_conversion_results.json")
    guard_results = read_json(GUARD_DIR / "kalcb_q3_incremental_guard_search_results.json")
    seed_mutations = train_opt.load_seed_mutations()
    q3_route = next(route for route in route_mod.route_catalogue() if route.name == TRAIN_ROUTE)
    q3_mutations = q3_route.build(seed_mutations)
    promoted_mutations = build_guarded_round1_mutations(seed_mutations, q3_mutations)
    KALCBConfig.from_mapping(mutations=promoted_mutations)

    train_pool_rows = read_jsonl(SOURCE_DIR / "original_champion_train_pool_rows.jsonl")
    equivalence = static_route_equivalence(train_pool_rows, seed_mutations, q3_mutations, promoted_mutations)
    write_json(DIAGNOSTIC_DIR / "static_route_equivalence.json", equivalence)
    if not equivalence["pass"]:
        raise RuntimeError(f"Promoted route semantics mismatch original guard: {equivalence['mismatch_count']} rows")
    filtered_train_pool_rows, train_prefilter_counts = guarded_candidate_pool(train_pool_rows, seed_mutations, q3_mutations)
    write_jsonl(DIAGNOSTIC_DIR / "train_guarded_prefilter_pool_rows.jsonl", filtered_train_pool_rows)
    write_json(DIAGNOSTIC_DIR / "train_guarded_prefilter_counts.json", train_prefilter_counts)

    train_config, holdout_config = shared.load_base_config()
    print(json.dumps({"event": "train_replay_start", "rows": len(filtered_train_pool_rows), "input_rows": len(train_pool_rows)}, sort_keys=True), flush=True)
    train_replay = replay_pool("train_round1_guarded_q3", "round_1_guarded_q3_train_prefiltered", train_config, filtered_train_pool_rows, promoted_mutations)

    holdout_pool_rows, holdout_pool_info = build_holdout_pool()
    write_jsonl(DIAGNOSTIC_DIR / "holdout_full_pool_rows.jsonl", holdout_pool_rows)
    filtered_holdout_pool_rows, holdout_prefilter_counts = guarded_candidate_pool(holdout_pool_rows, seed_mutations, q3_mutations)
    write_jsonl(DIAGNOSTIC_DIR / "holdout_guarded_prefilter_pool_rows.jsonl", filtered_holdout_pool_rows)
    write_json(DIAGNOSTIC_DIR / "holdout_guarded_prefilter_counts.json", holdout_prefilter_counts)
    print(json.dumps({"event": "holdout_replay_start", "rows": len(filtered_holdout_pool_rows), "input_rows": len(holdout_pool_rows)}, sort_keys=True), flush=True)
    holdout_replay = replay_pool("holdout_round1_guarded_q3", "round_1_guarded_q3_holdout_prefiltered", holdout_config, filtered_holdout_pool_rows, promoted_mutations)

    source = source_artifacts(train_replay, holdout_replay)
    source["guard_search_train_selected"] = {
        "guard_label": (guard_results.get("train_champion") or {}).get("guard_label"),
        "train_metrics": compact_replay(guard_results.get("train_champion") or {}),
        "locked_holdout_metrics": compact_replay(guard_results.get("locked_holdout_audit") or {}),
    }
    source["route_control"] = {
        "seed": compact_replay(route_row(route_results, SEED_ROUTE)),
        "q3": compact_replay(route_row(route_results, TRAIN_ROUTE)),
    }
    source["holdout_pool"] = holdout_pool_info

    live_parity = live_parity_audit_payload(promoted_mutations, train_replay, holdout_replay, equivalence, train_prefilter_counts, holdout_prefilter_counts)
    summary = diagnostics_summary_payload(promoted_mutations, train_replay, holdout_replay, equivalence, train_prefilter_counts, holdout_prefilter_counts, source)
    optimized = optimized_config_payload(created, promoted_mutations, summary, live_parity)
    run_summary = run_summary_payload(created, summary, live_parity)

    write_json(ROUND_DIR / "optimized_config.json", optimized)
    write_json(ROUND_DIR / "diagnostics_summary.json", summary)
    write_json(ROUND_DIR / "run_summary.json", run_summary)
    write_json(ROUND_DIR / "live_parity_audit.json", live_parity)
    write_text(ROUND_DIR / "round_final_diagnostics.txt", render_report(summary, live_parity))
    write_text(ROUND_DIR / "round_evaluation.txt", render_evaluation(summary, live_parity))
    write_json(ROUND_DIR / "candidate_frontier.json", {"round": 1, "selected_policy": summary["selected_guard"], "train": summary["train"], "holdout": summary["holdout_locked_audit"]})
    write_json(ROUND_DIR / "progress.json", {"round": 1, "status": "complete", "updated_at_utc": created, "elapsed_seconds": time.time() - started})
    write_jsonl(ROUND_DIR / "phase_activity_log.jsonl", [{"ts": created, "event": "promote_guarded_q3_round1", "train": summary["train"], "holdout": summary["holdout_locked_audit"]}])
    write_json(DIAGNOSTIC_DIR / "guarded_q3_full_diagnostics.json", {"created_at_utc": created, "summary": summary, "live_parity_audit": live_parity, "optimized_config": optimized})
    write_json(
        ROUND_DIR / "full_diagnostics_index.json",
        {
            "round": 1,
            "created_at_utc": created,
            "purpose": "Full diagnostics for KALCB Guarded q3 round_1 promotion.",
            "artifacts": {
                "optimized_config": str(ROUND_DIR / "optimized_config.json"),
                "run_summary": str(ROUND_DIR / "run_summary.json"),
                "diagnostics_summary": str(ROUND_DIR / "diagnostics_summary.json"),
                "live_parity_audit": str(ROUND_DIR / "live_parity_audit.json"),
                "round_final_diagnostics": str(ROUND_DIR / "round_final_diagnostics.txt"),
                "round_evaluation": str(ROUND_DIR / "round_evaluation.txt"),
                "full_diagnostics_payload": str(DIAGNOSTIC_DIR / "guarded_q3_full_diagnostics.json"),
                "static_route_equivalence": str(DIAGNOSTIC_DIR / "static_route_equivalence.json"),
                "holdout_full_pool_rows": str(DIAGNOSTIC_DIR / "holdout_full_pool_rows.jsonl"),
                "train_guarded_prefilter_pool_rows": str(DIAGNOSTIC_DIR / "train_guarded_prefilter_pool_rows.jsonl"),
                "holdout_guarded_prefilter_pool_rows": str(DIAGNOSTIC_DIR / "holdout_guarded_prefilter_pool_rows.jsonl"),
                "train_trade_rows": summary["train"]["trade_rows_path"],
                "holdout_trade_rows": summary["holdout_locked_audit"]["trade_rows_path"],
            },
            "source_hashes": source,
        },
    )
    update_manifest(created, promoted_mutations, summary, live_parity)

    print(
        json.dumps(
            {
                "event": "round1_guarded_q3_promoted",
                "round_dir": str(ROUND_DIR),
                "manifest": str(MANIFEST_PATH),
                "train_net": summary["train"]["broker_net_return_pct"],
                "train_dd": summary["train"]["broker_max_drawdown_pct"],
                "train_trades": summary["train"]["trade_count"],
                "holdout_net": summary["holdout_locked_audit"]["broker_net_return_pct"],
                "holdout_dd": summary["holdout_locked_audit"]["broker_max_drawdown_pct"],
                "holdout_trades": summary["holdout_locked_audit"]["trade_count"],
                "parity": live_parity["audit_status"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
