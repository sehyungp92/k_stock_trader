"""Evaluate a frozen OLR recipe on a declared available window, fail closed.

The new input generation must explicitly bind the validation contract. This
command never updates the selected recipe or enables paper/live orders.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from olr_rule_study import digest, forward_validation_policy, load_study, read, signature, write
    from olr_rule_analysis import bootstrap_difference, trade_statistics, trade_difference
except ModuleNotFoundError:
    from scripts.olr_rule_study import digest, forward_validation_policy, load_study, read, signature, write
    from scripts.olr_rule_analysis import bootstrap_difference, trade_statistics, trade_difference


def check_contract(contract, study, manifest, window):
    payload = dict(contract)
    expected = payload.pop("contract_hash")
    if signature(payload) != expected:
        raise ValueError("Validation contract changed")
    if contract["study_hash"] != study["study_hash"]:
        raise ValueError("Wrong parent study")
    if manifest["source_generation_hash"] != contract["source_generation_hash"]:
        raise ValueError("Execution source changed; require a reviewed contract revision")
    if signature(contract["mutations"]) != contract["mutations_hash"]:
        raise ValueError("Frozen recipe changed")
    if window["start"] < contract["earliest_eligible_kst_date"] or window["end"] < window["start"]:
        raise ValueError("Requested evaluation predates the recipe freeze")
    evidence = manifest.get("evidence", {})
    if evidence.get("validation_contract_hash") != expected:
        raise ValueError("New input generation is not bound to the reserved validation contract")
    end = evidence.get("shared_market_inputs", {}).get("data_end", "")
    if end < window["end"]:
        raise ValueError("Qualified input coverage does not reach the declared evaluation window")


def verified_contract(path):
    contract = read(path)
    payload = {k: v for k, v in contract.items() if k != "contract_hash"}
    if signature(payload) != contract["contract_hash"] or signature(contract["mutations"]) != contract["mutations_hash"]:
        raise ValueError("Frozen validation contract changed")
    return contract


def register_protocol(study_path, destination, recipe_path=None):
    """Freeze a usable revision without amending the original prospective study."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new protocol directory; existing evidence is immutable")
    study = load_study(study_path)
    recipe = read(recipe_path) if recipe_path else {"selected": "control", "mutations": study["cases"]["control"]["mutations"]}
    if recipe.get("study_hash", study["study_hash"]) != study["study_hash"]:
        raise ValueError("Recipe belongs to a different study")
    if recipe["mutations"] != study["cases"][recipe["selected"]]["mutations"]:
        raise ValueError("Recipe differs from the registered candidate")
    engine = Path(study["engine_root"])
    frozen = datetime.now(timezone.utc)
    contract = {
        "version": "olr-prospective-validation-v4", "frozen_at_utc": frozen.isoformat(),
        "earliest_eligible_kst_date": (frozen.astimezone(ZoneInfo("Asia/Seoul")).date() + timedelta(days=1)).isoformat(),
        **forward_validation_policy(), "study_hash": study["study_hash"],
        "selected": recipe["selected"], "mutations": recipe["mutations"],
        "mutations_hash": signature(recipe["mutations"]), "source_generation_hash": study["source_generation_hash"],
        "parent_input_manifest": study["input_manifest"],
        "frozen_inputs": {name: digest(engine / name) for name in ("config/universe_103.yaml", "config/olr/sector_map.yaml")},
        "execution_model": study["config"]["capacity_scenario"],
        "scenarios": {"nominal": {"capacity_scenario": study["config"]["capacity_scenario"], "additional_slippage_bps": 0.0},
                      "combined-stress": {"capacity_scenario": {"name": "combined-stress", "participation": .005}, "additional_slippage_bps": 10.0}},
        "primary_scenario": "nominal",
        "stress_decision": {"positive_net_return_required": True, "maximum_drawdown": .15, "minimum_profit_factor": 1.1},
        "decision": {"maximum_drawdown": .15, "minimum_profit_factor": 1.1, "positive_net_expectancy_required": True,
                     "nonlower_net_return_than_control": True, "minimum_mean_net_r_ratio_vs_control": .95},
        "comparison": "control versus cash, or selected challenger versus control; not market-adjusted alpha",
        "interim_policy": "Capture all sessions including no-trade and losing days. Disclose every evaluation window; progress alone cannot establish independent performance or change the frozen recipe.",
        "required_inputs": ["causal daily and 14:30 candidates", "qualified completed market bars and references", "native continuous account replay with complete daily equity"],
        "operational_readiness": "Broker-connected paper parity is separate from offline performance validation.",
        "historical_tail": "May 13 to July 10 independence unconfirmed; excluded from this prospective contract",
        "live_promotion": False,
    }
    contract["contract_hash"] = signature(contract)
    write(destination / "study.json", study)
    write(destination / "prospective_validation_contract.json", contract)
    snapshot_validator(destination, study["study_hash"])
    return contract


def snapshot_validator(destination, study_hash):
    files = {}
    for name in ("olr_prospective_validation.py", "olr_rule_study.py", "olr_rule_analysis.py"):
        target = destination / "prospective_harness" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name), target)
        files[name] = digest(target)
    write(destination / "prospective_validator_manifest.json", {
        "study_hash": study_hash, "files": files, **forward_validation_policy()})


def revise_protocol(contract_path, destination):
    """Remove the session-count gate in a new version, preserving frozen evidence."""
    contract_path, destination = Path(contract_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new protocol directory; existing evidence is immutable")
    original = verified_contract(contract_path)
    study = load_study(contract_path.parent / "study.json")
    if study["study_hash"] != original["study_hash"]:
        raise ValueError("Wrong parent study")
    contract = {k: v for k, v in original.items() if k not in {"contract_hash", "evaluation_sessions"}}
    contract.update(forward_validation_policy())
    contract.update(version="olr-prospective-validation-v4", parent_contract_hash=original["contract_hash"],
        policy_revision={"revised_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "User removed future-session counts as completion conditions; recipe, frozen start and performance integrity requirements are retained.",
            "original_contract_path": str(contract_path), "original_contract_sha256": digest(contract_path)},
        interim_policy="Retain all sessions and disclose every evaluation window. No outcome-based endpoint selection, recipe changes or automatic performance qualification.")
    contract["contract_hash"] = signature(contract)
    write(destination / "study.json", study)
    write(destination / "prospective_validation_contract.json", contract)
    for name in ("optimized_config.json", "validation_baseline.json"):
        source = contract_path.parent / name
        if source.exists():
            shutil.copyfile(source, destination / name)
    snapshot_validator(destination, study["study_hash"])
    return contract


def evaluation_days(contract, manifest, calendar):
    """Use the declared available window, without waiting for a target count."""
    start = date.fromisoformat(contract["earliest_eligible_kst_date"])
    end_text = manifest.get("evidence", {}).get("shared_market_inputs", {}).get("data_end")
    if not end_text:
        raise ValueError("Declare evaluation data_end in the input manifest")
    end = date.fromisoformat(end_text)
    if end < start:
        raise ValueError("No eligible observations in the declared evaluation window")
    days, cursor = [], start
    while cursor <= end:
        if not calendar.has_coverage(cursor):
            raise ValueError("Evaluation window lacks sourced calendar coverage; require a reference revision")
        if calendar.is_trading_day(cursor):
            if not calendar.has_session_coverage(cursor):
                raise ValueError("Evaluation window lacks sourced session coverage; require a reference revision")
            days.append(cursor)
        cursor += timedelta(days=1)
    if not days:
        raise ValueError("No eligible exchange sessions in the declared evaluation window")
    return days


def prospective_uncertainty(left, right):
    result = bootstrap_difference(left, right)
    # A circular sample no longer than one block cannot identify block uncertainty.
    result["support_status"] = "inconclusive" if result["sessions"] <= result["block_sessions"] else "available"
    result["interpretation"] = "Descriptive paired session-block uncertainty on the declared window. Repeated looks and endpoint selection are not corrected; a completed report is not proof of independent alpha."
    return result


def setup_status(contract_path, output, manifest_path=None, baseline_path=None):
    """A setup check does not require future observations or broker connectivity."""
    contract = verified_contract(contract_path)
    study = load_study(Path(contract_path).parent / "study.json")
    problems, pending = [], []
    if study["study_hash"] != contract["study_hash"]:
        raise ValueError("Wrong parent study")
    engine = Path(study["engine_root"])
    if baseline_path:
        from scripts.generate_olr_kalcb_artifacts import _load_strategy_configs
        runtime = _load_strategy_configs(baseline_path, ("OLR",))
        if runtime["OLR"]["mutations"] != contract["mutations"]:
            problems.append("runtime_recipe_differs_from_frozen_recipe")
    for name, expected in contract.get("frozen_inputs", {}).items():
        if digest(engine / name) != expected:
            problems.append("frozen_universe_or_sector_changed:" + name)
    if manifest_path:
        manifest = read(manifest_path)
        if manifest["source_generation_hash"] != contract["source_generation_hash"]:
            problems.append("source_revision_not_bound")
        if manifest.get("evidence", {}).get("validation_contract_hash") != contract["contract_hash"]:
            problems.append("input_generation_not_bound")
        engine = Path(manifest_path).resolve().parent / "files"
        end = manifest.get("evidence", {}).get("shared_market_inputs", {}).get("data_end", "")
    else:
        end = ""
    if end < contract["earliest_eligible_kst_date"]:
        pending.append("new_market_observations_not_yet_available")
    import yaml
    sessions = yaml.safe_load((engine / "kis_core/data/krx_sessions.yaml").read_text())
    if str(sessions["coverage_end"]) < contract["earliest_eligible_kst_date"]:
        pending.append("extend_sourced_session_and_suspension_reference_coverage")
    result = {"contract_hash": contract["contract_hash"], "setup_pass": not problems, "problems": problems,
              "pending_data_requirements": pending, "earliest_eligible_kst_date": contract["earliest_eligible_kst_date"],
              **forward_validation_policy(), "final_performance_evaluated": False,
              "paper_connection_required_for_offline_validation": False, "live_promotion": False}
    write(Path(output) / "setup_status.json", result)
    return result


def record_session(contract_path, session_root, output):
    """Append a sealed existing recorder capture; never turn dry-run fills into returns."""
    contract = verified_contract(contract_path)
    root, output = Path(session_root).resolve(), Path(output).resolve()
    manifest = read(root / "session_manifest.json")
    day = date.fromisoformat(manifest["trade_date"])
    now = datetime.now(timezone.utc)
    if day < date.fromisoformat(contract["earliest_eligible_kst_date"]) or day > now.astimezone(ZoneInfo("Asia/Seoul")).date():
        raise ValueError("Session is outside elapsed prospective dates")
    if manifest.get("hash_contract_status") != "sealed" or not manifest.get("expected_hashes_complete"):
        raise ValueError("Capture is not sealed; missing observations must remain visible")
    config = (manifest.get("strategy_configs") or {}).get("OLR", {})
    if signature(config.get("payload", {})) != contract["mutations_hash"]:
        raise ValueError("Captured runtime recipe differs from the freeze")
    from deployment.olr_kalcb.session_capture import REQUIRED_EXPECTED_HASH_GROUPS, session_hashes
    if not set(REQUIRED_EXPECTED_HASH_GROUPS).issubset(manifest.get("expected_hashes", {})):
        raise ValueError("Capture is missing required sealed hash groups")
    actual = session_hashes(root)
    if any(actual.get(k) != v for k, v in manifest["expected_hashes"].items()):
        raise ValueError("Captured session changed after sealing")
    files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}
    archive = output / "sessions" / day.isoformat()
    if (output / (day.isoformat() + ".json")).exists() or archive.exists():
        raise FileExistsError("Session already recorded; preserve it rather than overwrite")
    shutil.copytree(root, archive)
    if any(digest(archive / name) != value for name, value in files.items()):
        raise ValueError("Capture changed while archiving; retain failed attempt for investigation")
    archived_hashes = session_hashes(archive)
    if any(archived_hashes.get(k) != v for k, v in manifest["expected_hashes"].items()):
        raise ValueError("Archived capture differs from the sealed observations")
    record = {"contract_hash": contract["contract_hash"], "trade_date": day.isoformat(),
              "recorded_at_utc": now.isoformat(), "capture_root": str(archive), "source_capture_root": str(root), "files": files,
              "mode": manifest.get("mode"), "performance_evaluated": False,
              "scope": "operational capture; native account evaluation uses the declared available window"}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents duplicates, retrospective replacements and retries overwriting evidence.
    with (output / (day.isoformat() + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
    return record


def scenario_configuration(config, mutations, scenario):
    """Apply a registered execution stress without changing selection or the parent recipe."""
    from strategy_olr.config import OLRConfig
    result, recipe = copy.deepcopy(config), copy.deepcopy(mutations)
    extra = scenario.get("additional_slippage_bps", 0.0)
    if extra:
        slippage = OLRConfig.from_mapping(result, recipe).slippage_bps + extra
        recipe.update({"olr.cost.slippage_bps": slippage, "olr.robustness.slippage_bps": slippage})
    if scenario.get("capacity_scenario"):
        result["capacity_scenario"] = copy.deepcopy(scenario["capacity_scenario"])
    result["initial_mutations"] = recipe
    return result, recipe


def evaluate(contract_path, manifest_path, output, preflight_only):
    contract_path, manifest_path, output = map(lambda p: Path(p).resolve(), (contract_path, manifest_path, output))
    contract = read(contract_path)
    protocol = read(contract_path.parent / "prospective_validator_manifest.json")
    executing = {"olr_prospective_validation.py": Path(__file__),
                 "olr_rule_study.py": Path(sys.modules[read.__module__].__file__),
                 "olr_rule_analysis.py": Path(sys.modules[bootstrap_difference.__module__].__file__)}
    for filename, path in executing.items():
        if digest(path) != protocol["files"][filename]:
            raise ValueError("Reserved validation implementation changed")
    study = load_study(contract_path.parent / "study.json")
    if protocol["study_hash"] != study["study_hash"]:
        raise ValueError("Wrong prospective protocol study")
    manifest = read(manifest_path)
    # Check bindings before importing any new generation's code.
    preliminary = {"start": contract["earliest_eligible_kst_date"], "end": contract["earliest_eligible_kst_date"]}
    check_contract(contract, study, manifest, preliminary)
    engine = manifest_path.parent / "files"
    sys.path.insert(0, str(engine))
    os.chdir(engine)
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.core.input_manifest import InputManifest
    from kis_core.trading_calendar import get_trading_calendar
    if source_generation_hash() != contract["source_generation_hash"]:
        raise ValueError("Executing source identity differs from contract")
    generation = InputManifest(manifest_path)
    with generation.activate():
        calendar = get_trading_calendar()
        days = evaluation_days(contract, manifest, calendar)
    window = {"start": days[0].isoformat(), "end": days[-1].isoformat()}
    check_contract(contract, study, manifest, window)
    if preflight_only:
        print("Preflight passed", window, "sessions", len(days), flush=True)
        return
    if (output / "validation_request.json").exists():
        raise FileExistsError("Reserved evaluation already started; preserve it rather than overwrite")
    config = copy.deepcopy(study["config"])
    config.update(input_manifest_path=str(manifest_path), date_range=window, use_full_available_window=True)
    for key in ("holdout_start", "holdout_end", "regression_window"):
        config.pop(key, None)
    selected = contract["selected"]
    names = ["control"] if selected == "control" else ["control", selected]
    request = {"contract_hash": contract["contract_hash"], "manifest_sha256": digest(manifest_path),
               "window": window, "sessions": len(days), "cases": names,
               "scenarios": contract.get("scenarios", {"nominal": {}}), "validator_sha256": digest(__file__)}
    write(output / "validation_request.json", request)
    from types import SimpleNamespace
    from backtests.strategies.olr.replay_cache import load_olr_declared_window_bundle
    from backtests.strategies.olr.runner import run_olr_backtest
    from backtests.strategies.olr.final_diagnostics import export_case, json_value, release_diagnostic_replay_memory
    from backtests.analysis.metrics import session_equity
    scenario_summaries, scenario_daily = {}, {}
    nominal_trades = {}
    scenarios = contract.get("scenarios", {"nominal": {}})
    for scenario_name, scenario in scenarios.items():
        daily, summaries = {}, {}
        scenario_output = output if scenario_name == "nominal" else output / scenario_name
        for name in names:
            mutations = study["cases"][name]["mutations"] if name == "control" else contract["mutations"]
            case_config, mutations = scenario_configuration(config, mutations, scenario)
            bundle = load_olr_declared_window_bundle(case_config, mutations)
            release_diagnostic_replay_memory(SimpleNamespace(_result_cache={}, _last_result=None))
            result = run_olr_backtest(case_config, mutations, replay_bundle=bundle)
            export_case(result, case_config, mutations, scenario_output / name)
            replay = result.replay_result
            with generation.activate():
                values, coverage = session_equity(replay.equity_curve, replay.timestamps, replay.broker.mark_coverage,
                                                 declared_window=window, initial_cash_only=True)
            if coverage["status"] != "complete" or len(values) != len(days) + 1:
                raise ValueError("Reserved evaluation has incomplete session marks")
            daily[name] = [{"date": day.isoformat(), "net_return": b / a - 1}
                           for day, a, b in zip(days, values, values[1:])]
            write(scenario_output / name / "daily_returns.json", daily[name])
            trades = [json_value(t) for t in result.trades]
            summaries[name] = {"metrics": result.metrics, "trade_statistics": trade_statistics(trades),
                               "closed_trades_per_session": len(trades) / len(days)}
            if scenario_name == "nominal":
                nominal_trades[name] = trades
            del result, replay, bundle
            release_diagnostic_replay_memory(SimpleNamespace(_result_cache={}, _last_result=None))
        scenario_summaries[scenario_name], scenario_daily[scenario_name] = summaries, daily
    daily, summaries = scenario_daily["nominal"], scenario_summaries["nominal"]
    reference = daily["control"] if selected != "control" else [{**r, "net_return": 0} for r in daily[selected]]
    uncertainty = prospective_uncertainty(daily[selected], reference)
    if selected != "control":
        changes = trade_difference(nominal_trades[selected], nominal_trades["control"])
        write(output / "incremental_trade_attribution.json", {"trades": changes,
            "added_trade_statistics": trade_statistics(changes["added"]),
            "removed_trade_statistics": trade_statistics(changes["removed"])})
    metrics = summaries[selected]["metrics"]
    reasons = []
    if metrics["official_mtm_net_return_pct"] <= 0:
        reasons.append("nonpositive_net_return")
    if metrics["max_drawdown_pct"] > contract["decision"]["maximum_drawdown"]:
        reasons.append("drawdown")
    if metrics["profit_factor"] < contract["decision"]["minimum_profit_factor"]:
        reasons.append("profit_factor")
    if summaries[selected]["trade_statistics"]["mean_net_r"] is None or summaries[selected]["trade_statistics"]["mean_net_r"] <= 0:
        reasons.append("nonpositive_net_expectancy")
    if selected != "control" and metrics["total_trades"] < summaries["control"]["metrics"]["total_trades"]:
        reasons.append("lower_frequency_than_control")
    if selected != "control":
        control = summaries["control"]
        if metrics["official_mtm_net_return_pct"] < control["metrics"]["official_mtm_net_return_pct"]:
            reasons.append("lower_net_return_than_control")
        quality, prior_quality = summaries[selected]["trade_statistics"]["mean_net_r"], control["trade_statistics"]["mean_net_r"]
        if quality is not None and prior_quality is not None and quality < contract["decision"].get("minimum_mean_net_r_ratio_vs_control", .95) * prior_quality:
            reasons.append("lower_net_trade_quality_than_control")
    if uncertainty["support_status"] == "inconclusive" or uncertainty["conditional_95pct_interval"][0] <= 0:
        reasons.append("incremental_evidence_inconclusive")
    stress_policy = contract.get("stress_decision", {})
    for scenario_name, accounts in scenario_summaries.items():
        if scenario_name == "nominal":
            continue
        stressed = accounts[selected]["metrics"]
        if (stressed["official_mtm_net_return_pct"] <= 0 or
            stressed["max_drawdown_pct"] > stress_policy["maximum_drawdown"] or
            stressed["profit_factor"] < stress_policy["minimum_profit_factor"]):
            reasons.append(scenario_name + ":economic_robustness")
    write(output / "validation_result.json", {"request": request, "cases": summaries, "scenarios": scenario_summaries,
        "paired_uncertainty": uncertainty, "stress_is_independent_sample": False,
        "evaluation_complete": True, **forward_validation_policy(),
        "performance_gates_pass": not reasons, "reasons": reasons, "paper_live_parity": "not_established_by_simulation",
        "live_promotion": False, "status": "reserved_simulation_complete_review_required",
        "reference": "frozen control" if selected != "control" else "cash; not a market-adjusted alpha benchmark"})
    print("Reserved validation complete; performance gates", not reasons, "paper/live parity remains separate", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--register-study")
    parser.add_argument("--revise-contract", help="Create a new protocol version without a session-count completion gate")
    parser.add_argument("--recipe")
    parser.add_argument("--baseline-manifest")
    parser.add_argument("--record-session")
    args = parser.parse_args()
    if args.revise_contract:
        result = revise_protocol(args.revise_contract, args.output)
        print("Revised", result["contract_hash"], result["earliest_eligible_kst_date"])
    elif args.register_study:
        result = register_protocol(args.register_study, args.output, args.recipe)
        print("Registered", result["contract_hash"], result["earliest_eligible_kst_date"])
    elif not args.contract:
        parser.error("--contract is required outside registration")
    elif args.preflight_only:
        print(json.dumps(setup_status(args.contract, args.output, args.manifest, args.baseline_manifest), indent=2))
    elif args.record_session:
        print(json.dumps(record_session(args.contract, args.record_session, args.output), indent=2))
    elif not args.manifest:
        parser.error("--manifest is required for final evaluation")
    else:
        evaluate(args.contract, args.manifest, args.output, False)


if __name__ == "__main__":
    main()
