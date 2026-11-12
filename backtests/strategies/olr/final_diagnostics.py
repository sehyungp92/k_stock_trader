"""Observer-only final reports, invoked by the shared PhaseRunner lifecycle."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path

from backtests.auto.shared.phase_state import _atomic_write_json
from backtests.engine.contracts import execution_valid
from backtests.engine.evaluation_identity import request_identity
from backtests.auto.shared.cache_keys import stable_signature


def json_value(value):
    if hasattr(value, "to_json_dict"):
        return json_value(value.to_json_dict())
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_value(row), sort_keys=True, default=str) + "\n")


def require_bound_result(result, config, mutations):
    if not execution_valid(result.metrics, expected_request=request_identity(config, mutations),
                           expected_mutations=stable_signature(mutations)):
        raise ValueError("Final OLR diagnostics require a current, passing, request-bound execution receipt")
    if result.metrics.get("daily_equity_coverage", {}).get("status") != "complete":
        raise ValueError("Final OLR diagnostics require complete declared-window equity marks")


def release_diagnostic_replay_memory(plugin):
    """Drop completed replay references and rebuildable caches, never active data.

    The caller retains its current bundle. Clearing cache dictionaries does not
    mutate that bundle, its bars, snapshots, exported results or audit receipts.
    """
    from backtests.core import input_manifest
    from . import replay_cache, research_sweep
    plugin._result_cache.clear()
    plugin._last_result = None
    for name in ("_BUNDLE_CACHE", "_DATASET_CACHE", "_RESEARCH_SNAPSHOT_CACHE",
                 "_STAGE1_SNAPSHOT_CACHE", "_AFTERNOON_CONTEXT_CACHE"):
        getattr(replay_cache, name).clear()
    research_sweep._FRAME_CACHE.clear()
    input_manifest._VERIFIED_BYTES.clear()
    gc.collect()


def export_case(result, config, mutations, directory):
    """Do not run strategy transitions, re-audit, or alter any replay object."""
    require_bound_result(result, config, mutations)
    directory.mkdir(parents=True, exist_ok=True)
    replay = result.replay_result
    broker = replay.broker
    metrics = dict(result.metrics)
    _atomic_write_json({"config": config, "mutations": mutations,
                        "evaluation_request": request_identity(config, mutations)}, directory / "request.json")
    _atomic_write_json(metrics, directory / "metrics.json")
    _atomic_write_json(metrics.get("audit_receipt"), directory / "audit_receipt.json")
    write_rows(directory / "trades.jsonl.gz", result.trades)
    write_rows(directory / "fills.jsonl.gz", broker.fills)
    write_rows(directory / "decisions.jsonl.gz", result.decisions)
    write_rows(directory / "ledger.jsonl.gz", broker.journal)
    write_rows(directory / "equity.jsonl.gz", ({"timestamp": ts, "equity": value, "mark_coverage": coverage}
        for ts, value, coverage in zip(replay.timestamps, replay.equity_curve[1:], broker.mark_coverage, strict=True)))
    groups = defaultdict(list)
    for trade in result.trades:
        rank = trade.route_metadata.get("candidate_rank", trade.cohort_metadata.get("rank", "unknown"))
        for dimension, label in (("entry_month", trade.entry_fill_time.strftime("%Y-%m")),
                                 ("exit_reason", trade.exit_reason), ("symbol", trade.symbol), ("rank", rank)):
            groups[(dimension, str(label))].append(trade)
    cohorts = []
    for (dimension, label), trades in sorted(groups.items()):
        gross = sum(t.gross_pnl for t in trades)
        net = sum(t.net_pnl for t in trades)
        risk = [float(t.route_metadata.get("risk_per_share", 0) or 0) * t.qty for t in trades]
        net_r = [t.net_pnl / r for t, r in zip(trades, risk) if r > 0]
        cohorts.append({"dimension": dimension, "label": label, "closed_entry_count": len(trades),
            "wins": sum(t.net_pnl > 0 for t in trades), "gross_pnl_krw": gross, "net_pnl_krw": net,
            "explicit_fees_krw": gross - net, "mean_net_r": sum(net_r) / len(net_r) if net_r else None,
            "net_r_denominator_count": len(net_r)})
    _atomic_write_json(cohorts, directory / "cohorts.json")
    return {"metrics": metrics, "path": str(directory), "cohorts": cohorts}


def write_full_diagnostics(plugin, state, output_dir, *, round_num, round_name):
    # Replay and sealed-export validation must request exactly the same cases.
    from .exported_diagnostics import diagnostic_requests
    from .replay_cache import load_olr_real_replay_bundle, load_olr_declared_window_bundle
    from .runner import run_olr_backtest
    from .plugin import _format_olr_diagnostics, _greedy_from_state

    mutations = dict(state.cumulative_mutations)
    config = dict(plugin.training_config)
    baseline = plugin._evaluate_result(mutations)
    training_text = _format_olr_diagnostics(plugin.num_phases, baseline.metrics,
        _greedy_from_state(state, baseline.metrics), plugin.execution_context)
    cases = {"training": export_case(baseline, config, mutations, output_dir / "diagnostics" / "training")}
    print("Final diagnostics: training replay and ledger exported", flush=True)
    plugin._result_cache.clear()
    del baseline
    gc.collect()
    if plugin.capability_level.lower() not in {"synthetic", "dry_run"}:
        requests = diagnostic_requests(config, mutations, plugin.config.get("regression_window"))
        bundle = load_olr_real_replay_bundle(config, mutations)
        for name, request in requests.items():
            if name == "training":
                continue
            case_config, case_mutations = request["config"], request["mutations"]
            if name == "consumed_regression":
                del bundle
                release_diagnostic_replay_memory(plugin)
                bundle = load_olr_declared_window_bundle(case_config, case_mutations)
            release_diagnostic_replay_memory(plugin)
            print(f"Final diagnostics: replaying {name}", flush=True)
            result = run_olr_backtest(case_config, case_mutations, replay_bundle=bundle)
            cases[name] = export_case(result, case_config, case_mutations, output_dir / "diagnostics" / name)
            del result
            gc.collect()
        del bundle
        gc.collect()
    return render_full_diagnostics(plugin, state, output_dir, round_num=round_num,
        round_name=round_name, cases=cases, training_text=training_text)


def render_full_diagnostics(plugin, state, output_dir, *, round_num, round_name, cases, training_text):
    """Format already validated native exports without executing strategy transitions."""
    mutations = dict(state.cumulative_mutations)
    config = dict(plugin.training_config)
    limitations = [
        "Final diagnostics replay the saved recipe; this refresh performs no new parameter search. Prior qualification is recorded separately.",
        "April-May 2026 has already been consumed. Regression results are not an untouched out-of-sample claim.",
        "Paper/live execution qualification and a newly reserved untouched window remain required before deployment.",
        "Capacity is a causal research participation assumption; observed queue liquidity and live fill parity are not established.",
        "Static 103-symbol universe: universe selection/survivorship sensitivity remains unquantified.",
        "Gross MFE and daily outcome labels are hindsight/filter-quality diagnostics, not benchmark-relative alpha or promotion evidence.",
        "No-upward-compounding caps allocation equity at initial capital; losses still reduce available sizing and cash constraints remain active.",
        "Open terminal inventory is marked to market with no fabricated liquidation; closed-trade metrics exclude it.",
    ]
    # Native capacity studies can have a different declared feed/window. Their
    # adapter must state that scope rather than inherit an unrelated OOS claim.
    limitations = list(getattr(plugin, "diagnostic_limitations", limitations))
    payload = {"strategy": "olr", "round": round_num, "round_name": round_name,
        "selected_candidate": plugin.config.get("selected_candidate") or round_name or plugin.config.get("baseline_name", "OLR saved recipe"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_status": plugin.config.get("promotion_status", "research_only_consumed_regression_paper_live_pending"),
        "development_qualification": plugin.config.get("development_qualification"),
        "reporting_source": getattr(plugin, "reporting_source", None),
        "mutations": mutations, "cases": cases, "limitations": limitations,
        "training_window": config.get("date_range"), "regression_window": plugin.config.get("regression_window"),
        "input_manifest_path": config.get("input_manifest_path")}
    payload["diagnostic_scope"] = getattr(plugin, "diagnostic_scope", None)
    # Every exporter uses the same analytical layer. Native interval adapters
    # supply the sealed source plan; generic exports declare missing coverage.
    from .analytical_diagnostics import build_analysis
    analytical, overview = build_analysis(cases, output_dir, getattr(plugin, 'analytical_plan', None), config.get('initial_equity'))
    payload['analytical_diagnostics'] = analytical
    text = [f"# OLR round {round_num}: {round_name}", "", overview, '',
            '## Execution contract and accepted controls', '', training_text, "", "Validated replay cases (returns and drawdowns in percent):"]
    for name, case in cases.items():
        m = case["metrics"]
        text.append(f"{name}: MTM return {100*m['official_mtm_net_return_pct']:.8f}%; DD {100*m['max_drawdown_pct']:.8f}%; "
                    f"closed trades {m['total_trades']:.0f}; entries {m.get('entry_lifecycle_count')}; PF {m['profit_factor']:.6f}; "
                    f"daily Sharpe {m['sharpe']:.6f}; audit {m.get('audit_status')}")
        text.append("  Equity coverage: " + json.dumps(m.get("daily_equity_coverage"), sort_keys=True))
    text.extend(["", "Interpretation and remaining limitations:", *limitations])
    for name, artifact in analytical['cases'].items():
        text.extend(['', '---', '', f'Full account report: {name}', '',
                     (output_dir / artifact['report']).read_text(encoding='utf-8').replace(
                         '](analysis.json)', f']({artifact["json"]})')])
    report = "\n".join(text) + "\n"
    payload["final_diagnostics_text_sha256"] = hashlib.sha256(report.encode()).hexdigest()
    _atomic_write_json(payload, output_dir / "round_final_full_diagnostics.json")
    # The receipt hashes UTF-8 LF bytes; Windows must not translate them to CRLF.
    (output_dir / "round_final_diagnostics.txt").write_text(report, encoding="utf-8", newline="\n")
    (output_dir / "round_final_diagnostics.md").write_text(report, encoding="utf-8", newline="\n")
    _atomic_write_json({"selected": payload}, output_dir / "optimized_results.json")
    return payload
