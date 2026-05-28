from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_olr.config import OLR_CORE_VERSION

from .allocation_sweep import (
    DEFAULT_OUTPUT_DIR as ALLOCATION_OUTPUT_DIR,
    _decision_key,
    _fill_key,
    _key_summary,
    _order_key,
    _position_key,
)
from .research_sweep import DEFAULT_HOLDOUT_DAYS
from .runner import attach_overnight_labels_to_snapshots, compile_olr_replay_bundle, run_olr_backtest
from .trade_plan_sweep import CandidateSource, build_compiled_execution_set


ALLOCATION_HOLDOUT_EVAL_VERSION = "olr-allocation-holdout-official-mtm-v1"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/olr/allocation_holdout")


def run_allocation_holdout_eval(
    config: dict[str, Any] | None = None,
    *,
    research_sweep_path: str | Path | None = None,
    allocation_sweep_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 5,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    max_workers: int = 2,
    max_stress_scenarios: int = 80,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stress_grid = _auction_stress_grid()[: max(0, int(max_stress_scenarios))]
    if dry_run and allocation_sweep_path is None:
        payload = {
            "strategy": "olr",
            "dry_run": True,
            "research_sweep_path": str(research_sweep_path or ""),
            "allocation_sweep_path": "",
            "top_n": max(1, int(top_n)),
            "stress_scenarios": len(stress_grid),
            "max_workers": min(max(1, int(max_workers)), 2),
            "official_performance": False,
            "requires_allocation_sweep_artifact_for_execution": True,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload
    allocation_path = Path(allocation_sweep_path) if allocation_sweep_path else _latest_allocation_sweep_path()
    allocation_payload = json.loads(allocation_path.read_text(encoding="utf-8"))
    research_path = Path(research_sweep_path or allocation_payload.get("research_sweep_path") or "")
    if not research_path.exists():
        raise FileNotFoundError(f"OLR research sweep artifact not found: {research_path}")
    research_payload = json.loads(research_path.read_text(encoding="utf-8"))
    finalist_source = "top_official_train" if allocation_payload.get("top_official_train") else "top_train"
    finalist_rows = list(allocation_payload.get(finalist_source) or [])[: max(1, int(top_n))]
    sources = tuple(_source_from_row(row, rank=index + 1) for index, row in enumerate(finalist_rows))
    if dry_run:
        payload = {
            "strategy": "olr",
            "dry_run": True,
            "research_sweep_path": str(research_path),
            "allocation_sweep_path": str(allocation_path),
            "finalist_source": finalist_source,
            "top_n": len(finalist_rows),
            "stress_scenarios": len(stress_grid),
            "max_workers": min(max(1, int(max_workers)), 2),
            "official_performance": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload

    holdout_config = _holdout_config(dict(config or {}), holdout_days)
    compiled_full = build_compiled_execution_set(
        holdout_config,
        research_payload,
        sources,
        holdout_days=holdout_days,
        use_fast_cache=False,
        include_holdout=True,
    )
    holdout_dates = tuple(day for day in compiled_full.eligible_dates if day >= compiled_full.dataset.holdout_start)
    if not holdout_dates:
        raise ValueError("OLR allocation holdout evaluation found no eligible holdout sessions with a next-session exit")
    audit_rows = []
    stress_rows = []
    for row, source in zip(finalist_rows, sources):
        allocation = dict(row.get("allocation") or {})
        mutations = _execution_mutations(row)
        fast_bundle = _bundle_for_source(compiled_full, source, holdout_dates, candidate_only=True)
        full_bundle = _bundle_for_source(compiled_full, source, holdout_dates, candidate_only=False)
        fast = run_olr_backtest({**holdout_config, "capability_level": "compiled"}, mutations, replay_bundle=fast_bundle)
        full = run_olr_backtest({**holdout_config, "capability_level": "compiled"}, mutations, replay_bundle=full_bundle)
        audit_rows.append(_audit_row(row, source, allocation, fast, full))
        for stress in stress_grid:
            stress_result = run_olr_backtest(
                {**holdout_config, "capability_level": "compiled"},
                {**mutations, **stress["mutations"]},
                replay_bundle=full_bundle,
            )
            stress_rows.append(
                {
                    "candidate": source.name,
                    "allocation": allocation.get("name", ""),
                    "scenario": stress["name"],
                    "official_mtm_net_return_pct": stress_result.metrics.get("official_mtm_net_return_pct", 0.0),
                    "official_mtm_max_drawdown_pct": stress_result.metrics.get("official_mtm_max_drawdown_pct", 0.0),
                    "auction_nonfill_count": stress_result.metrics.get("auction_nonfill_count", 0.0),
                }
            )
    payload = {
        "strategy": "olr",
        "eval_version": ALLOCATION_HOLDOUT_EVAL_VERSION,
        "strategy_core_version": OLR_CORE_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_performance": False,
        "research_sweep_path": str(research_path),
        "allocation_sweep_path": str(allocation_path),
        "finalist_source": finalist_source,
        "holdout_policy": {
            "holdout_days": int(holdout_days),
            "holdout_start": compiled_full.dataset.holdout_start.isoformat(),
            "train_dates_excluded": True,
            "selection_uses_holdout": False,
            "allocation_optimized_on_holdout": False,
        },
        "causality_policy": {
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "intraday_selection_cutoff": "timestamp < 14:30 KST",
            "execution_path": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker",
        },
        "stress_policy": {
            "base_slippage_bps": [5, 10, 20, 35, 50],
            "auction_adverse_bps": [0, 10, 25, 50],
            "auction_nonfill_rates": [0.0, 0.05, 0.10, 0.20],
            "limit_offset_bps": [0, 10, 25, 50],
            "deterministic_nonfill": True,
            "scenarios_evaluated": len(stress_grid),
        },
        "source_fingerprints": {
            "research_sweep_hash": str(research_payload.get("sweep_hash") or ""),
            "allocation_sweep_hash": str(allocation_payload.get("sweep_hash") or ""),
            "compiled_full": compiled_full.source_fingerprint,
            "candidate_artifacts": compiled_full.candidate_artifact_hash,
            "daily_intraday": compiled_full.dataset.source_fingerprint,
        },
        "metric_contract": {
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm",
            "promotion_requires_audit_pass": True,
            "official_replay_pass": bool(audit_rows),
            "audit_pass": False,
            "audit_status": "pending",
            "official_metrics": ["official_mtm_net_return_pct", "official_mtm_max_drawdown_pct", "official_mtm_sharpe"],
            "proxy_metrics": [],
        },
        "execution_contract": {
            "strategy": "olr",
            "phase_framework_version": "custom-olr-allocation-holdout",
            "strategy_core_version": OLR_CORE_VERSION,
            "source_fingerprint": compiled_full.source_fingerprint,
            "feature_manifest_hash": compiled_full.dataset.source_fingerprint,
            "candidate_snapshot_hash": compiled_full.candidate_artifact_hash,
            "date_window": {
                "start": holdout_dates[0].isoformat() if holdout_dates else "",
                "end": holdout_dates[-1].isoformat() if holdout_dates else "",
                "sessions": len(holdout_dates),
            },
            "fill_timing": "close_auction_to_next_close",
            "auction_mode": "close_auction",
            "capability_level": "compiled",
            "replay_mode": "full_holdout_official_mtm",
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "OLRReplayAdapter -> OLR core -> neutral actions -> SimBroker.equity_curve_bar_level_mtm",
        },
        "holdout_window": {
            "start": holdout_dates[0].isoformat() if holdout_dates else "",
            "end": holdout_dates[-1].isoformat() if holdout_dates else "",
            "sessions": len(holdout_dates),
        },
        "audits": audit_rows,
        "stress": stress_rows,
    }
    payload["audit_pass"] = all(row.get("audit_pass") for row in audit_rows)
    payload["metric_contract"]["audit_pass"] = payload["audit_pass"]
    payload["metric_contract"]["audit_status"] = "audited_full_bundle_passed" if payload["audit_pass"] else "audit_failed"
    payload["eval_hash"] = stable_signature(
        {
            "version": ALLOCATION_HOLDOUT_EVAL_VERSION,
            "holdout_window": payload["holdout_window"],
            "audits": audit_rows,
            "stress_count": len(stress_rows),
        }
    )
    json_path = out / f"olr_allocation_holdout_{payload['eval_hash'][:12]}.json"
    md_path = out / f"olr_allocation_holdout_{payload['eval_hash'][:12]}.md"
    seed_path = out / f"olr_allocation_holdout_seed_{payload['eval_hash'][:12]}.json"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path), "seed": str(seed_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    seed_path.write_text(json.dumps(_seed_payload(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _bundle_for_source(compiled, source: CandidateSource, holdout_dates: tuple, *, candidate_only: bool):
    snapshots = {
        day: snapshot
        for day, snapshot in compiled.snapshots_by_source.get(source.name, {}).items()
        if day in set(holdout_dates)
    }
    snapshots = attach_overnight_labels_to_snapshots(snapshots, compiled.dataset.overnight_labels_by_key)
    symbols_by_day = {
        day: {candidate.symbol for candidate in snapshot.candidates}
        for day, snapshot in snapshots.items()
    }
    allowed_symbols_by_day = {day: set(symbols) for day, symbols in symbols_by_day.items()}
    for day, symbols in symbols_by_day.items():
        next_day = compiled.next_session_by_date.get(day)
        if next_day is not None:
            allowed_symbols_by_day.setdefault(next_day, set()).update(symbols)
    bars = []
    needed_dates = set(holdout_dates)
    needed_dates.update(compiled.next_session_by_date.get(day) for day in holdout_dates if compiled.next_session_by_date.get(day) is not None)
    for key, day_bars in compiled.dataset.bars_by_key.items():
        day, symbol = key
        if day not in needed_dates:
            continue
        if candidate_only and symbol not in allowed_symbols_by_day.get(day, set()):
            continue
        bars.extend(day_bars)
    return compile_olr_replay_bundle(
        bars=bars,
        snapshots=snapshots,
        source_fingerprint=stable_signature([compiled.source_fingerprint, source.name, candidate_only, [day.isoformat() for day in holdout_dates]]),
    )


def _holdout_config(config: dict[str, Any], holdout_days: int) -> dict[str, Any]:
    out = dict(config or {})
    out["holdout_days"] = int(holdout_days)
    out["use_full_available_window"] = True
    return out


def _source_from_row(row: dict[str, Any], *, rank: int) -> CandidateSource:
    raw = dict(row.get("source") or {})
    return CandidateSource(
        rank=int(raw.get("rank", rank) or rank),
        name=str(raw.get("name") or row.get("name") or f"source_{rank}"),
        stage1_name=str(raw.get("stage1_name") or ""),
        stage2_name=str(raw.get("stage2_name") or raw.get("name") or ""),
        score=float(raw.get("score", row.get("score", 0.0)) or 0.0),
        mutations=dict(raw.get("mutations") or {}),
        artifact_hash=str(raw.get("artifact_hash") or ""),
    )


def _execution_mutations(row: dict[str, Any]) -> dict[str, Any]:
    allocation = dict(row.get("allocation") or {})
    trade_plan = dict(row.get("trade_plan") or {})
    mutations = _allocation_mutations(allocation)
    if isinstance(trade_plan.get("entry"), dict):
        mutations["olr.trade_plan.entry"] = dict(trade_plan["entry"])
    if isinstance(trade_plan.get("exit"), dict):
        mutations["olr.trade_plan.exit"] = dict(trade_plan["exit"])
    return mutations


def _allocation_mutations(allocation: dict[str, Any]) -> dict[str, Any]:
    return {
        "olr.allocation.mode": allocation.get("mode", "selected_equal_capped"),
        "olr.allocation.target_gross_exposure": allocation.get("target_gross_exposure", 1.0),
        "olr.allocation.max_position_pct": allocation.get("max_position_pct", 0.25),
        "olr.allocation.rank_decay": allocation.get("rank_decay", 1.0),
        "olr.allocation.min_selected": allocation.get("min_selected", 1),
    }


def _audit_row(row, source, allocation, fast, full) -> dict[str, Any]:
    metric_keys = ("official_mtm_net_return_pct", "official_mtm_max_drawdown_pct", "official_mtm_sharpe")
    deltas = {
        key: float(full.metrics.get(key, 0.0) or 0.0) - float(fast.metrics.get(key, 0.0) or 0.0)
        for key in metric_keys
    }
    trade_hash_fast = stable_signature([trade.to_json_dict() for trade in fast.trades])
    trade_hash_full = stable_signature([trade.to_json_dict() for trade in full.trades])
    fast_keys = _official_audit_key_evidence(fast)
    full_keys = _official_audit_key_evidence(full)
    keys_match = _audit_keys_match(fast_keys, full_keys)
    audit_pass = (
        trade_hash_fast == trade_hash_full
        and keys_match
        and max((abs(value) for value in deltas.values()), default=0.0) <= 1e-10
    )
    return {
        "name": row.get("name") or source.name,
        "source": asdict(source),
        "allocation": allocation,
        "fast_metrics": {key: fast.metrics.get(key, 0.0) for key in metric_keys},
        "full_metrics": {key: full.metrics.get(key, 0.0) for key in metric_keys},
        "metric_deltas": deltas,
        "fast_trade_hash": trade_hash_fast,
        "full_trade_hash": trade_hash_full,
        "official_audit_keys_match": keys_match,
        "fast_official_audit_keys": fast_keys,
        "full_official_audit_keys": full_keys,
        "audit_pass": audit_pass,
    }


def _official_audit_key_evidence(result) -> dict[str, Any]:
    decisions = [
        decision
        for decision in result.decisions
        if decision.strategy_id == "OLR" and str(decision.decision_code).endswith("_SUBMITTED")
    ]
    selected_candidate_keys = []
    for decision in decisions:
        meta = dict(getattr(decision, "metadata", {}) or {})
        selected_candidate_keys.append(
            "|".join(
                [
                    str(getattr(decision, "timestamp", "")).split(" ")[0],
                    str(getattr(decision, "symbol", "")).zfill(6),
                    f"rank={int(meta.get('candidate_rank', 0) or 0)}",
                    f"artifact={meta.get('source_artifact_hash', '')}",
                ]
            )
        )
    broker = result.replay_result.broker
    return {
        "selected_candidate_keys": _key_summary(selected_candidate_keys),
        "submitted_order_keys": _key_summary(_decision_key(decision) for decision in decisions),
        "fill_keys": _key_summary(_fill_key(fill) for fill in broker.fills if fill.strategy_id == "OLR"),
        "rejected_order_keys": _key_summary(_order_key(order) for order in broker.rejected_orders if order.strategy_id == "OLR"),
        "nonfill_order_keys": _key_summary(
            _order_key(order)
            for order in broker.expired_orders
            if order.strategy_id == "OLR" and order.order_type == "CLOSE_AUCTION"
        ),
        "open_order_keys": _key_summary(_order_key(order) for order in broker.orders if order.strategy_id == "OLR"),
        "open_position_keys": _key_summary(_position_key(position) for position in broker.positions.values() if position.strategy_id == "OLR"),
    }


def _audit_keys_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    key_names = (
        "selected_candidate_keys",
        "submitted_order_keys",
        "fill_keys",
        "rejected_order_keys",
        "nonfill_order_keys",
        "open_order_keys",
        "open_position_keys",
    )
    return all(
        dict(left.get(name) or {}).get("hash") == dict(right.get(name) or {}).get("hash")
        for name in key_names
    )


def _auction_stress_grid() -> list[dict[str, Any]]:
    rows = []
    for slip in (5, 10, 20, 35, 50):
        for adverse in (0, 10, 25, 50):
            for nonfill in (0.0, 0.05, 0.10, 0.20):
                for offset in (0, 10, 25, 50):
                    rows.append(
                        {
                            "name": f"slip{slip}_adv{adverse}_nf{int(nonfill*100)}_off{offset}",
                            "mutations": {
                                "olr.cost.slippage_bps": slip,
                                "olr.execution.auction_adverse_bps": adverse,
                                "olr.execution.auction_nonfill_rate": nonfill,
                                "olr.execution.auction_limit_offset_bps": offset,
                            },
                        }
                    )
    return rows


def _latest_allocation_sweep_path() -> Path:
    files = sorted(ALLOCATION_OUTPUT_DIR.glob("olr_allocation_sweep_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No OLR allocation sweep artifacts found under {ALLOCATION_OUTPUT_DIR}")
    return files[0]


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Allocation Holdout Audit",
        f"- Eval hash: `{payload.get('eval_hash')}`",
        f"- Holdout: {payload['holdout_window']['start']} to {payload['holdout_window']['end']} ({payload['holdout_window']['sessions']} sessions)",
        f"- Official performance: `{payload.get('official_performance')}`",
        f"- Audit pass: `{payload.get('audit_pass')}`",
        "",
        "| Rank | Candidate | Fast MTM | Full MTM | Delta | Pass |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(payload.get("audits", []), start=1):
        fast = row.get("fast_metrics", {}).get("official_mtm_net_return_pct", 0.0)
        full = row.get("full_metrics", {}).get("official_mtm_net_return_pct", 0.0)
        delta = row.get("metric_deltas", {}).get("official_mtm_net_return_pct", 0.0)
        lines.append(f"| {rank} | {row.get('name')} | {100*fast:.2f}% | {100*full:.2f}% | {100*delta:.4f}% | {row.get('audit_pass')} |")
    return "\n".join(lines) + "\n"


def _seed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best = (payload.get("audits") or [{}])[0]
    return {
        "strategy": "olr",
        "source_eval_hash": payload.get("eval_hash"),
        "official_performance": False,
        "promoted_starting_baseline": best,
        "policy": "Holdout audit seed only; promotion still requires paper/live parity evidence.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OLR allocation holdout audit through OLR core + SimBroker.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--research-sweep-path", default=None)
    parser.add_argument("--allocation-sweep-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-stress-scenarios", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = normalize_runtime_config("olr", load_yaml_config(args.config))
    payload = run_allocation_holdout_eval(
        cfg,
        research_sweep_path=args.research_sweep_path,
        allocation_sweep_path=args.allocation_sweep_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        holdout_days=args.holdout_days,
        max_workers=args.max_workers,
        max_stress_scenarios=args.max_stress_scenarios,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps({"eval_hash": payload["eval_hash"], "artifact_paths": payload["artifact_paths"], "audit_pass": payload["audit_pass"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
