"""Preregistered OLR rule ablations using a separately frozen native engine.

The evaluator never selects on the consumed regression window and never writes
canonical rounds. Its own bytes are frozen separately from the execution engine.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

RULES = "olr.afternoon.score_band_rules"
WEIGHTS = ("olr.afternoon.weight_sector_rotation", "olr.afternoon.weight_stock_sector_leadership")


def forward_validation_policy():
    return {
        "completion_requires_future_sessions": False,
        "evaluation_window_policy": "Evaluate all eligible sessions from the frozen start through the declared input data_end; no fixed session-count minimum.",
        "evidence_policy": "Report observed sample size and uncertainty. Completed implementation or evaluation does not establish independent performance or authorize deployment.",
    }


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def recipes(base):
    """Four admission policies crossed with current/zero two soft weights."""
    out = {}
    for gate in ("current", "all_middle_sectors", "tails_only", "no_bands"):
        for soft in (True, False):
            name = "control" if gate == "current" and soft else gate + ("_soft" if soft else "_zero")
            mutations = copy.deepcopy(base)
            if gate == "all_middle_sectors":
                mutations[RULES] = [r for r in mutations[RULES] if "sector_admission" not in r]
                for rule in mutations[RULES]:
                    rule.pop("allowed_sectors", None)
            elif gate == "tails_only":
                mutations[RULES] = [r for r in mutations[RULES] if not ("min_score" in r and "max_score" in r)]
            elif gate == "no_bands":
                mutations[RULES] = []
                mutations["olr.afternoon.reject_score_min"] = 0.0
                mutations["olr.afternoon.reject_score_max"] = 0.0
            if not soft:
                for key in WEIGHTS:
                    mutations[key] = 0.0
            out[name] = {"gate": gate, "soft_weights": soft, "mutations": mutations}
    assert len(out) == len({signature(r["mutations"]) for r in out.values()}) == 8
    return out


def stability_recipes(base):
    """Local diagnostic neighbours; never additional selectable finalists."""
    out = {}
    for factor in (0.5, 1.5):
        mutations = copy.deepcopy(base)
        for key in WEIGHTS:
            mutations[key] = float(base.get(key, 0)) * factor
        if mutations != base:
            out[f"weights_x{factor}"] = mutations
    for boundary in ("low", "middle", "high"):
        for shift in (-25, 25):
            mutations = copy.deepcopy(base)
            for rule in mutations.get(RULES, []):
                if boundary == "low" and "max_score" in rule and "min_score" not in rule:
                    rule["max_score"] += shift
                elif boundary == "middle" and "min_score" in rule and "max_score" in rule:
                    rule["min_score"] += shift
                    rule["max_score"] += shift
                elif boundary == "high" and "min_score" in rule and "max_score" not in rule:
                    rule["min_score"] += shift
            if mutations != base:
                out[f"{boundary}_{shift:+d}"] = mutations
    return out


def register(round_dir, destination):
    round_dir, destination = Path(round_dir).resolve(), Path(destination).resolve()
    if (destination / "study.json").exists():
        raise FileExistsError("Study already registered; resume it instead")
    request = read(round_dir / "diagnostics/training/request.json")
    manifest_path = Path(request["config"]["input_manifest_path"]).resolve()
    manifest = read(manifest_path)
    base = request["mutations"]
    assert base.get("olr.shadow_reranker.enabled", False) is False
    harness = destination / "harness" / Path(__file__).name
    harness.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, harness)
    config = request["config"]
    # Enables the existing, checksum-verified dataset cache; no policy change.
    config["corrective_study_hash"] = "olr-rule-ablation-v1"
    spec = {
        "version": "olr-rule-ablation-v1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_dir": str(round_dir), "output": str(destination),
        "input_manifest": str(manifest_path), "input_manifest_file_sha256": digest(manifest_path),
        "engine_root": str(manifest_path.parent / "files"),
        "source_generation_hash": manifest["source_generation_hash"],
        "harness": str(harness), "harness_sha256": digest(harness),
        "config": config, "cases": recipes(base),
        "baseline_metrics": read(round_dir / "diagnostics/training/metrics.json"),
        "baseline_metrics_sha256": digest(round_dir / "diagnostics/training/metrics.json"),
        "selection_scope": "training_only; consumed regression cannot select or rank recipes",
        "selection_policy": {
            "return_and_closed_trades_not_below_control": "nominal AND sizing_cap",
            "minimum_mean_net_r_ratio": 0.95,
            "maximum_drawdown": 0.12,
            "maximum_drawdown_increase": 0.01,
            "minimum_profit_factor": 1.1,
            "strict_gain_required": True,
            "ranking": "largest minimum relative terminal-wealth gain across nominal and sizing cap; then trades; then name",
            "maximum_challengers": 1,
            "fallback": "control",
        },
        "stress_policy": {
            "scenarios": ["participation-stress", "cost-stress", "combined-stress"],
            "minimum_return_ratio_vs_control": 0.90,
            "minimum_trade_count_ratio_vs_control": 0.95,
            "maximum_drawdown": 0.15,
            "minimum_profit_factor": 1.1,
            "failure_action": "retain control; no replacement search",
        },
        "stability_policy": {
            "control_neighbours": stability_recipes(base),
            "challenger_neighbours": "same recipe generator applied to frozen challenger",
            "selection": "diagnostic neighbours never promoted",
            "veto": "challenger median neighbour return < 90% of parent or minimum < 60%; drawdown >15%",
        },
        "independent_evaluation": {
            "existing_holdout": "consumed: 2026-04-01 through 2026-05-12",
            "later_historical_dates": "independence and qualified input coverage unverified; not opened by this study",
            "policy": "freeze one recipe before genuinely unused data or prospective evaluation",
            "adaptive_reselection": False,
        },
        "limitations": ["historical design hindsight", "static 103-symbol universe", "research capacity assumption", "paper/live parity pending"],
    }
    spec["study_hash"] = signature(spec)
    write(destination / "study.json", spec)
    support = {"version": "olr-rule-support-v1", "registered_at_utc": spec["registered_at_utc"],
               "study_hash": spec["study_hash"], "minimum_changed_selected_issuer_days": 10,
               "minimum_changed_calendar_months": 3,
               "support_definition": "Changed selected membership or rank; score-only changes do not count",
               "selection_scope": "training candidate decisions only"}
    support["policy_hash"] = signature(support)
    write(destination / "support_policy.json", support)
    pipeline_files = {}
    for filename in ("olr_rule_analysis.py", "olr_rule_finish.py", "olr_rule_alpha_attribution.py", "olr_rule_resolved_pipeline.py"):
        target = destination / "harness" / filename
        shutil.copyfile(Path(__file__).with_name(filename), target)
        pipeline_files[filename] = digest(target)
    write(destination / "analysis_pipeline_manifest.json", {"registered_at_utc": spec["registered_at_utc"],
           "study_hash": spec["study_hash"], "files": pipeline_files,
           "resolution_worker": "olr_rule_resolved_pipeline.py"})
    future_files = {}
    for filename in ("olr_prospective_validation.py", "olr_rule_study.py", "olr_rule_analysis.py"):
        target = destination / "prospective_harness" / filename
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(filename), target)
        future_files[filename] = digest(target)
    write(destination / "prospective_validator_manifest.json", {"registered_at_utc": spec["registered_at_utc"],
           "study_hash": spec["study_hash"], "files": future_files,
           **forward_validation_policy(), "block_sessions": 10, "confidence_interval": 0.95,
           "criterion": "Lower paired mean-daily-return interval above zero, positive net expectancy, PF >=1.1, DD <=15%; selected challenger frequency no lower than control"})
    print(json.dumps({"study": str(destination / "study.json"), "cases": len(spec["cases"]), "hash": spec["study_hash"]}), flush=True)


def load_study(path):
    spec = read(path)
    expected = spec.pop("study_hash")
    if signature(spec) != expected:
        raise ValueError("Registered study bytes changed")
    spec["study_hash"] = expected
    if digest(spec["harness"]) != spec["harness_sha256"]:
        raise ValueError("Frozen evaluator changed")
    if digest(spec["input_manifest"]) != spec["input_manifest_file_sha256"]:
        raise ValueError("Input manifest changed")
    return spec


def candidate_rows(snapshots):
    for day, snapshot in sorted(snapshots.items()):
        yield {"date": day.isoformat(), "selected": [
            {"symbol": c.symbol, "sector": c.sector, "rank": c.rank, "score": c.selection_score,
             "rule": c.metadata.get("afternoon_score_band_rule"),
             "features": c.metadata.get("afternoon_features", {})}
            for c in snapshot.candidates],
            "eligible_pool_count": snapshot.metadata.get("afternoon_candidate_pool_count"),
            "rejected": snapshot.metadata.get("afternoon_rejected_symbols", {})}


def export_or_preserve_failure(result, config, mutations, out, export_case, write_rows, json_value):
    """A rejected result retains forensic evidence, never a completion receipt."""
    try:
        return export_case(result, config, mutations, out)
    except Exception as exc:
        write(out / "failure.json", {"status": "rejected_not_qualified",
              "error_type": type(exc).__name__, "error": str(exc),
              "config": config, "mutations": mutations,
              "metrics": json_value(result.metrics),
              "scope": "Forensic only; do not score or publish rejected economics as performance"})
        replay = result.replay_result
        for label, records in (("trades", result.trades), ("fills", replay.broker.fills),
                               ("ledger", replay.broker.journal), ("decisions", result.decisions)):
            write_rows(out / ("failed_" + label + ".jsonl.gz"), records)
        raise


def replay_worker(spec, name, mode, mutation_file=None):
    root = Path(spec["engine_root"])
    sys.path.insert(0, str(root))
    os.chdir(root)
    temp = Path(spec["output"]) / "temp"
    temp.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp)
    import tempfile
    tempfile.tempdir = str(temp)
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.olr.replay_cache import load_olr_real_replay_bundle, load_olr_declared_window_bundle
    from backtests.strategies.olr.runner import run_olr_backtest, snapshots_from_bundle
    from backtests.strategies.olr.final_diagnostics import export_case, write_rows, release_diagnostic_replay_memory, json_value
    from backtests.analysis.metrics import session_equity
    from kis_core.trading_calendar import get_trading_calendar
    from backtests.core.input_manifest import InputManifest
    if source_generation_hash() != spec["source_generation_hash"]:
        raise ValueError("Execution source drift")
    mutations = read(mutation_file) if mutation_file else spec["cases"][name]["mutations"]
    if mutation_file:
        registered = [r["mutations"] for r in spec["cases"].values()]
        allowed = registered + [m for base in registered for m in stability_recipes(base).values()]
        if signature(mutations) not in {signature(m) for m in allowed}:
            raise ValueError("Unregistered candidate or stability neighbour")
    config = copy.deepcopy(spec["config"])
    if mode == "regression":
        config["date_range"] = dict(config["regression_window"])
        config["use_full_available_window"] = True
        config.pop("holdout_start", None)
        config.pop("holdout_end", None)
    elif mode == "sizing_cap":
        mutations = {**mutations, "olr.allocation.equity_cap_krw": float(config["initial_equity"])}
    elif mode in {"participation-stress", "cost-stress", "combined-stress"}:
        from backtests.engine.capacity import CAPACITY_SCENARIOS
        from strategy_olr.config import OLRConfig
        _, participation, bps = next(r for r in CAPACITY_SCENARIOS if r[0] == mode)
        cfg = OLRConfig.from_mapping(config, mutations)
        config["capacity_scenario"] = {"name": mode, "participation": participation}
        mutations = {**mutations, "olr.cost.slippage_bps": cfg.slippage_bps + bps,
                     "olr.robustness.slippage_bps": cfg.slippage_bps + bps}
    elif mode != "nominal":
        raise ValueError("Unknown replay mode")
    out = Path(spec["output"]) / "cases" / name / mode
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "completed.json"
    job = {"study_hash": spec["study_hash"], "case": name, "mode": mode,
           "mutations": mutations, "config": config}
    if receipt.exists():
        if read(receipt)["job_hash"] != signature(job):
            raise ValueError("Completed job identity mismatch")
        print("Already completed", name, mode, flush=True)
        return
    write(out / "job.json", job)
    print(datetime.now(timezone.utc).isoformat(), "Building causal candidates", name, mode, flush=True)
    loader = load_olr_declared_window_bundle if mode == "regression" else load_olr_real_replay_bundle
    bundle = loader(config, mutations)
    write_rows(out / "candidate_selection.jsonl.gz", candidate_rows(snapshots_from_bundle(bundle)))
    release_diagnostic_replay_memory(SimpleNamespace(_result_cache={}, _last_result=None))
    print(datetime.now(timezone.utc).isoformat(), "Native replay and bound audit", name, mode, flush=True)
    result = run_olr_backtest(config, mutations, replay_bundle=bundle)
    export_or_preserve_failure(result, config, mutations, out, export_case, write_rows, json_value)
    replay = result.replay_result
    generation = InputManifest(spec["input_manifest"])
    with generation.activate():
        values, coverage = session_equity(replay.equity_curve, replay.timestamps, replay.broker.mark_coverage,
                                         declared_window=config["date_range"], initial_cash_only=True)
        calendar = get_trading_calendar()
        days = []
        cursor = date.fromisoformat(config["date_range"]["start"])
        while cursor <= date.fromisoformat(config["date_range"]["end"]):
            if calendar.is_trading_day(cursor):
                days.append(cursor.isoformat())
            cursor += timedelta(days=1)
    assert coverage["status"] == "complete" and len(values) == len(days) + 1
    write(out / "daily_returns.json", [
        {"date": day, "equity": now, "net_return": now / prior - 1, "net_pnl": now - prior}
        for day, prior, now in zip(days, values, values[1:])])
    metrics = result.metrics
    if name == "control" and mode == "nominal":
        for key in ("official_mtm_net_return_pct", "max_drawdown_pct", "total_trades", "profit_factor"):
            if abs(float(metrics[key]) - float(spec["baseline_metrics"][key])) > 1e-9:
                raise ValueError("Control economic parity failed: " + key)
    if source_generation_hash() != spec["source_generation_hash"]:
        raise ValueError("Execution source changed during replay")
    outputs = {p.name: digest(p) for p in out.iterdir()
               if p.is_file() and p.name not in {"completed.json", "worker.log"}}
    write(receipt, {"job_hash": signature(job), "study_hash": spec["study_hash"], "source_hash": spec["source_generation_hash"],
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "outputs": outputs,
                    "headline": {k: metrics[k] for k in ("official_mtm_net_return_pct", "max_drawdown_pct", "total_trades", "profit_factor")}})
    print("PASSED", name, mode, json.dumps(read(receipt)["headline"]), flush=True)
    del result, replay, bundle
    gc.collect()


def run_core(spec, study_path):
    # Separate processes bound memory and isolate each account. Resume only
    # completed, content-verified jobs. Control parity runs before challengers.
    names = ["control"] + sorted(n for n in spec["cases"] if n != "control")
    for mode in ("nominal", "sizing_cap"):
        for name in names:
            out = Path(spec["output"]) / "cases" / name / mode
            completed = out / "completed.json"
            if completed.exists():
                receipt = read(completed)
                if receipt["study_hash"] != spec["study_hash"]:
                    raise ValueError("Wrong study result")
                for filename, expected in receipt["outputs"].items():
                    if digest(out / filename) != expected:
                        raise ValueError("Completed output changed")
                print("Verified cached", name, mode, flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            print("START", name, mode, flush=True)
            with (out / "worker.log").open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-B", "-u", spec["harness"], "worker", "--study", str(study_path),
                                "--case", name, "--mode", mode], cwd=spec["engine_root"], stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            print("DONE", name, mode, json.dumps(read(completed)["headline"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("register")
    init.add_argument("--round-dir", required=True)
    init.add_argument("--output", required=True)
    for command in ("worker", "core"):
        cli = sub.add_parser(command)
        cli.add_argument("--study", type=Path, required=True)
        if command == "worker":
            cli.add_argument("--case", required=True)
            cli.add_argument("--mode", required=True)
            cli.add_argument("--mutation-file")
    args = parser.parse_args()
    if args.command == "register":
        register(args.round_dir, args.output)
    else:
        spec = load_study(args.study)
        if args.command == "worker":
            replay_worker(spec, args.case, args.mode, args.mutation_file)
        else:
            run_core(spec, args.study)


if __name__ == "__main__":
    main()
