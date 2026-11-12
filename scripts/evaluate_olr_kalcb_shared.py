from __future__ import annotations

# ruff: noqa: E402 -- repository scripts add the project root before local imports.

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtests.config import load_yaml_config
from backtests.core.input_manifest import InputManifest, active_manifest, read_input_bytes
from backtests.core.shared_market_inputs import compare_consumed_bars, market_input_contract
from backtests.strategies.portfolio_synergy.source_replay import (
    PromotedReplayInputs,
    load_promoted_kalcb_replay,
    load_promoted_olr_snapshots,
    overlap_attribution,
    require_parity,
    run_kalcb_standalone,
    run_native_risk_shared_replay,
    run_olr_standalone,
    shared_vs_standalone_attribution,
    summarize_replay,
    _load_pinned_parquet_bars,
)
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot


PINNED_INTRADAY_END = "20260512"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay promoted KALCB + OLR through their native cores on one shared account."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "config/olr_kalcb/shared_replay.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostic-paired", action="store_true",
        help="Compare legacy/corrected clocks; never assert historical qualification")
    parser.add_argument("--diagnostic-cache", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--verify-inputs-only", action="store_true",
        help="Verify both retained input paths and emit receipts without evaluating performance")
    parser.add_argument("--archived-feed-scope", action="store_true",
        help="Diagnostic reproduction only: use the original candidate-filtered feed")
    parser.add_argument("--diagnostic-target", choices=("KALCB", "OLR", "shared"),
        help="Evaluate one corrected account at a time to bound replay memory")
    parser.add_argument("--native-policy-grid", type=Path,
        help="Frozen JSON experiment list; only supported causal native policy fields are accepted")
    parser.add_argument("--capacity-sensitivity", action="store_true",
        help="Run the four declared participation/cost scenarios, one account at a time")
    from backtests.engine.capacity import CAPACITY_SCENARIOS
    parser.add_argument("--capacity-scenario", choices=tuple(row[0] for row in CAPACITY_SCENARIOS),
        help="Run one registered capacity scenario; use a single diagnostic target for bounded memory")
    parser.add_argument("--slippage-sensitivity", action="store_true",
        help="Additionally replay +5/+10/+25 bps adverse slippage on the shared account")
    args = parser.parse_args()
    if args.archived_feed_scope and not (args.diagnostic_paired or args.verify_inputs_only):
        parser.error("Archived feed scope is only available for diagnostic evidence")
    if args.diagnostic_target and not args.diagnostic_paired:
        parser.error("A single target requires --diagnostic-paired and remains unqualified")
    if (args.capacity_sensitivity or args.capacity_scenario) and not args.diagnostic_paired:
        parser.error("Capacity research requires --diagnostic-paired and a new output path")
    if (args.capacity_sensitivity or args.capacity_scenario) and (args.native_policy_grid or args.slippage_sensitivity):
        parser.error("Capacity study already includes registered cost stress; run other studies separately")
    if args.native_policy_grid and (not args.input_manifest or not args.diagnostic_paired):
        parser.error("Native policy experiments require a frozen manifest and --diagnostic-paired")
    if args.input_manifest is None:
        raw_spec = json.loads(args.spec.read_text(encoding="utf-8"))
        configured = raw_spec.get("input_manifest_path")
        if not configured:
            raise ValueError("Shared replay requires one declared input manifest for both strategies")
        args.input_manifest = ROOT / configured
    if args.input_manifest:
        manifest = InputManifest(args.input_manifest)
        from backtests.auto.shared.cache_keys import source_generation_hash
        if manifest.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Frozen research generation does not match current source")
        with manifest.activate():
            return run(args)
    return run(args)


def run(args):
    if (args.diagnostic_paired or args.verify_inputs_only) and (args.output is None or args.output.exists()):
        raise ValueError("Input verification and paired diagnostics require a new --output path to preserve existing evidence")
    input_root = active_manifest().root if active_manifest() else ROOT
    spec = _read_json(_repo_path(args.spec))
    contract = market_input_contract()
    if contract is None:
        raise ValueError("Shared replay requires a shared market input contract")
    initial_equity = float(spec["initial_equity"])
    cache_dir = active_manifest().path.parent / "derived_artifacts" / "shared_replay"
    manifest_before = None
    if args.diagnostic_paired:
        from backtests.auto.shared.cache_keys import file_content_hash, source_generation_hash
        manifest_paths = {_repo_path(args.spec), Path(__file__), input_root / "kis_core/data/krx_holidays.yaml"}
        for sid in ("kalcb", "olr"):
            for key in ("optimization_config", "optimized_result", "market_bar_digest"):
                if key in spec[sid]:
                    manifest_paths.add(_repo_path(spec[sid][key]))
            manifest_paths.update(_repo_path(spec[sid]["candidate_snapshot_root"]).glob("*.json"))
        manifest_paths.update((input_root / "data/kis_intraday_parquet").glob(f"*/*_5m_*_{PINNED_INTRADAY_END}.parquet"))
        for symbol, selected_end in contract.get("snapshot_end_overrides", {}).items():
            manifest_paths.update((input_root / "data/kis_intraday_parquet" / symbol).glob(f"{symbol}_5m_*_{selected_end}.parquet"))
        manifest_paths.update(_repo_path(spec["olr"]["daily_data_root"]).rglob("*.parquet"))
        def manifest():
            return {"source_generation": source_generation_hash(),
                "files": {str(p): file_content_hash(p) for p in sorted(manifest_paths)}}
        manifest_before = manifest()
        cache_dir = args.diagnostic_cache or args.output.with_suffix(".input-cache")

    kalcb_spec = dict(spec["kalcb"])
    kalcb_raw = load_yaml_config(_repo_path(kalcb_spec["optimization_config"]))
    kalcb_optimized = _read_json(_repo_path(kalcb_spec["optimized_result"]))
    kalcb_config, kalcb_replay, kalcb_lineage = load_promoted_kalcb_replay(
        input_root,
        kalcb_raw,
        kalcb_optimized,
        cache_dir=cache_dir,
        snapshot_root=_repo_path(kalcb_spec["candidate_snapshot_root"]),
        intraday_root=input_root / "data/kis_intraday_parquet",
        data_snapshot_end=PINNED_INTRADAY_END,
        market_bar_digest_path=_repo_path(kalcb_spec["market_bar_digest"]),
        snapshot_end_overrides=contract.get("snapshot_end_overrides"),
        allow_digest_mismatch=bool(args.diagnostic_paired or args.verify_inputs_only),
    )

    olr_spec = dict(spec["olr"])
    olr_raw = load_yaml_config(_repo_path(olr_spec["optimization_config"]))
    olr_optimized = _read_json(_repo_path(olr_spec["optimized_result"]))
    olr_mutations = dict(olr_optimized.get("mutations") or {})
    olr_config = OLRConfig.from_mapping(olr_raw, olr_mutations)
    olr_snapshots, olr_identity = load_promoted_olr_snapshots(
        _repo_path(olr_spec["candidate_snapshot_root"]),
        daily_data_root=_repo_path(olr_spec["daily_data_root"]),
        training_end=str((olr_optimized.get("train_window") or {}).get("date_end") or ""),
        expected_candidate_snapshot_hash=str(
            (olr_optimized.get("execution_contract") or {}).get("candidate_snapshot_hash") or ""
        ),
        expected_candidate_config_hash=str(olr_spec["candidate_config_hash"]),
        round_generated_at_utc=str(olr_spec["round_generated_at_utc"]),
    )
    olr_pairs = _olr_replay_pairs(olr_snapshots, kalcb_replay.session_dates)
    archived_scope = getattr(args, "archived_feed_scope", False)
    if not archived_scope:
        # Continue market observations after removal from the admission watchlist.
        # No future candidates are exposed early: each symbol starts on its first
        # admission date, then remains observable through the requested window.
        kalcb_pairs = management_replay_pairs(kalcb_replay.snapshots, kalcb_replay.session_dates)
        kalcb_replay = replace(kalcb_replay, bars=tuple(_load_pinned_bars(kalcb_pairs)))
        olr_pairs = management_replay_pairs(olr_snapshots, kalcb_replay.session_dates)
        kalcb_lineage["archived_bar_count"] = kalcb_lineage["bars"]
        kalcb_lineage["bars"] = len(kalcb_replay.bars)
    kalcb_lineage["feed_scope"] = "archived_candidate_filter" if archived_scope else "management_through_window_end"
    olr_bars = tuple(_load_pinned_bars(olr_pairs))
    olr_lineage = {
        "round": int(olr_optimized.get("round") or 0),
        "bars": len(olr_bars),
        "feed_scope": kalcb_lineage["feed_scope"],
        **olr_identity,
    }
    alignment = compare_consumed_bars(kalcb_replay.bars, olr_bars)
    common_identity = {"input_generation": active_manifest().identity,
                       "market_input_hash": contract["market_input_hash"],
                       "input_manifest_path": str(active_manifest().path)}
    for lineage in (kalcb_lineage, olr_lineage):
        lineage.update(common_identity)
    alignment.update(common_identity)
    alignment["status"] = "content_verified_and_aligned"
    alignment["historical_qualification"] = False
    if args.output:
        alignment["receipts"] = _write_input_receipts(args.output, alignment,
            {"KALCB": kalcb_replay.snapshots, "OLR": olr_snapshots})
    if args.verify_inputs_only:
        from backtests.auto.shared.cache_keys import source_generation_hash
        if active_manifest().payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Source changed during input verification; freeze a new generation")
        output = {"status": "inputs_verified_performance_not_evaluated", "input_alignment": alignment,
                  "lineage": {"KALCB": kalcb_lineage, "OLR": olr_lineage}}
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": output["status"], **common_identity}))
        return

    if getattr(args, "diagnostic_target", None) or args.capacity_sensitivity or getattr(args, "capacity_scenario", None):
        inputs = PromotedReplayInputs(kalcb_config=kalcb_config, kalcb_replay=kalcb_replay,
            olr_config=olr_config, olr_snapshots=olr_snapshots, olr_bars=olr_bars,
            lineage={"KALCB": kalcb_lineage, "OLR": olr_lineage})
        if args.capacity_sensitivity or getattr(args, "capacity_scenario", None):
            return run_capacity_study(args, inputs, spec, alignment, olr_optimized, manifest_before, manifest)
        return run_single_diagnostic(args, inputs, spec, alignment, olr_optimized, manifest_before, manifest)

    from backtests.engine.capacity import capacity_from_config
    # Paired clock diagnostics vary only the clock. Capacity has its own study.
    capacity_policy = None if args.diagnostic_paired else capacity_from_config(spec)
    kalcb_standalone = run_kalcb_standalone(kalcb_config, kalcb_replay, capacity_policy=capacity_policy)
    kalcb_summary = summarize_replay(kalcb_standalone, initial_equity=kalcb_replay.initial_equity)
    if not args.diagnostic_paired:
        require_parity(kalcb_summary, kalcb_spec["expected"], strategy="KALCB")

    olr_initial_equity = float((olr_optimized.get("execution_contract") or {}).get("initial_equity") or 10_000_000.0)
    olr_standalone = run_olr_standalone(
        olr_config,
        olr_snapshots,
        olr_bars,
        initial_equity=olr_initial_equity,
        capacity_policy=capacity_policy,
    )
    olr_summary = summarize_replay(olr_standalone, initial_equity=olr_initial_equity)
    if not args.diagnostic_paired:
        require_parity(olr_summary, olr_spec["expected"], strategy="OLR")

    inputs = PromotedReplayInputs(
        kalcb_config=kalcb_config,
        kalcb_replay=kalcb_replay,
        olr_config=olr_config,
        olr_snapshots=olr_snapshots,
        olr_bars=olr_bars,
        lineage={"KALCB": kalcb_lineage, "OLR": olr_lineage},
    )
    shared = run_native_risk_shared_replay(inputs, initial_equity=initial_equity, capacity_policy=capacity_policy)
    shared_summary = summarize_replay(shared, initial_equity=initial_equity)
    from backtests.engine.integrity import audit_broker
    from backtests.auto.shared.cache_keys import stable_signature
    generation = active_manifest().receipt() if active_manifest() else None
    for label, result, configuration in (("KALCB", kalcb_standalone, kalcb_config),
                                        ("OLR", olr_standalone, olr_config), ("shared", shared, spec)):
        audit_broker(result.broker, identity={"request": stable_signature([label, repr(configuration), alignment["strategies"], active_manifest().semantic_identity() if active_manifest() else None]),
            "mutations": stable_signature(kalcb_optimized.get("mutations") if label == "KALCB" else olr_mutations if label == "OLR" else spec),
            "input_manifest": generation, "configuration": repr(configuration), "lineage": inputs.lineage})
    if args.diagnostic_paired:
        from backtests.engine.contracts import LEGACY_EXECUTION_VERSION
        from backtests.engine.diagnostics import paired_report
        legacy_kalcb = run_kalcb_standalone(kalcb_config, kalcb_replay, execution_clock=LEGACY_EXECUTION_VERSION)
        legacy_olr = run_olr_standalone(olr_config, olr_snapshots, olr_bars,
            initial_equity=olr_initial_equity, execution_clock=LEGACY_EXECUTION_VERSION)
        legacy_shared = run_native_risk_shared_replay(inputs, initial_equity=initial_equity,
            execution_clock=LEGACY_EXECUTION_VERSION)
        output = {"status": "diagnostic_only_unqualified", "lineage": inputs.lineage,
            "input_alignment": alignment,
            "summaries": {"KALCB": asdict(kalcb_summary), "OLR": asdict(olr_summary), "shared": asdict(shared_summary)},
            "saved_evidence_status": "superseded_pending_selection_and_qualification",
            "capacity_scope": "disabled_on_both_sides_of_clock_comparison",
            "KALCB": paired_report(legacy_kalcb, kalcb_standalone),
            "OLR": paired_report(legacy_olr, olr_standalone),
            "shared": paired_report(legacy_shared, shared)}
        output["registered_capacity_thresholds"] = [.01, .1, 1.0]
        # Keep completed baseline evidence if a later sensitivity replay fails.
        output["input_manifest"] = manifest_before
        output["sensitivity_status"] = "running" if args.capacity_sensitivity or args.slippage_sensitivity else "not_requested"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
        if getattr(args, "native_policy_grid", None):
            from backtests.strategies.portfolio_synergy.native_evaluator import NativePortfolioPolicyEvaluator
            from backtests.auto.shared.types import Experiment
            from backtests.auto.shared.plugin_utils import CachedBatchEvaluator
            grid = json.loads(read_input_bytes(args.native_policy_grid))
            experiments = [Experiment(row["name"], row["mutations"]) for row in grid]
            if not experiments or len({e.name for e in experiments}) != len(experiments):
                raise ValueError("Native policy grid requires unique named experiments")
            evaluator = CachedBatchEvaluator(NativePortfolioPolicyEvaluator(inputs, initial_equity=initial_equity, configuration=spec))
            output["native_policy_experiments"] = [asdict(row) for row in evaluator(experiments, {})]
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
        if args.slippage_sensitivity:
            output["registered_additional_slippage_bps"] = [5, 10, 25]
            output["slippage_sensitivity"] = {}
            for extra in output["registered_additional_slippage_bps"]:
                stressed = replace(inputs,
                    kalcb_config=replace(kalcb_config, slippage_bps=kalcb_config.slippage_bps + extra),
                    olr_config=replace(olr_config, slippage_bps=olr_config.slippage_bps + extra))
                result = run_native_risk_shared_replay(stressed, initial_equity=initial_equity, capacity_policy=capacity_policy)
                output["slippage_sensitivity"][str(extra)] = asdict(summarize_replay(result, initial_equity=initial_equity))
        output["input_manifest"] = manifest_before
        output["sensitivity_status"] = "complete" if args.capacity_sensitivity or args.slippage_sensitivity else "not_requested"
        output["inputs_unchanged_during_run"] = manifest_before == manifest()
        if not output["inputs_unchanged_during_run"]:
            output["status"] = "invalid_input_generation_changed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": output["status"]}))
        return
    from backtests.engine.contracts import require_execution_valid
    for result in (kalcb_standalone, olr_standalone, shared):
        require_execution_valid(audit_broker(result.broker))
    output = {
        "schema_version": spec["schema_version"],
        "input_alignment": alignment,
        "result_basis": "exact_promoted_consumed_content_native_strategy_risk_one_shared_account",
        "standalone_parity": {
            "status": "pass",
            "KALCB": asdict(kalcb_summary),
            "OLR": asdict(olr_summary),
        },
        "shared": asdict(shared_summary),
        "shared_vs_standalone": shared_vs_standalone_attribution(
            shared_summary,
            {"KALCB": kalcb_summary, "OLR": olr_summary},
        ),
        "overlap": overlap_attribution(shared),
        "risk_contract": {
            "KALCB_native_intraday_leverage": float(kalcb_config.intraday_leverage),
            "OLR_native_intraday_leverage": 1.0,
            "combined_gross_leverage": max(float(kalcb_config.intraday_leverage), 1.0),
            "additional_portfolio_symbol_or_sector_caps": False,
        },
        "olr_comparison_resolution": {
            "status": olr_identity["identity_status"],
            "historical_profit_factor": float(olr_spec["expected"]["profit_factor"]),
            "current_reproduced_profit_factor": olr_summary.profit_factor,
            "strategy_decay_attribution_allowed": olr_identity["strategy_decay_attribution_allowed"],
        },
        "lineage": inputs.lineage,
        "pinned_intraday_snapshot_end": PINNED_INTRADAY_END,
    }
    output_path = args.output or _repo_path(spec["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def run_capacity_study(args, inputs, spec, alignment, olr_optimized, manifest_before, check_manifest):
    """Use the existing native replay paths, releasing each account before the next."""
    import gc
    from backtests.engine.capacity import BarParticipationScenario, CAPACITY_SCENARIOS, CAPACITY_VERSION
    from backtests.engine.integrity import audit_broker
    from backtests.auto.shared.cache_keys import stable_signature
    targets = (args.diagnostic_target,) if args.diagnostic_target else ("OLR", "KALCB", "shared")
    scenarios = [row for row in CAPACITY_SCENARIOS if not getattr(args, "capacity_scenario", None)
        or row[0] == args.capacity_scenario]
    output = {"status": "running", "model_version": CAPACITY_VERSION,
        "scope": "saved_recipes_under_declared_research_capacity; affected_selection_not_replaced",
        "historical_qualification": False, "phase_liquidity_verified": False,
        "input_alignment": alignment, "input_manifest": manifest_before, "accounts": {},
        "registered_scenarios": [{"name": n, "participation": p, "extra_slippage_bps_per_side": b}
                                 for n, p, b in scenarios]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
    save()
    for target in targets:
        output["accounts"][target] = {}
        for name, participation, extra in scenarios:
            print(json.dumps({"capacity_account": target, "scenario": name, "status": "running"}), flush=True)
            policy = BarParticipationScenario(participation, name)
            stressed = replace(inputs,
                kalcb_config=replace(inputs.kalcb_config, slippage_bps=inputs.kalcb_config.slippage_bps + extra),
                olr_config=replace(inputs.olr_config, slippage_bps=inputs.olr_config.slippage_bps + extra))
            initial = float(spec["initial_equity"])
            if target == "KALCB":
                initial = float(inputs.kalcb_replay.initial_equity)
                result = run_kalcb_standalone(stressed.kalcb_config, stressed.kalcb_replay, capacity_policy=policy)
            elif target == "OLR":
                initial = float((olr_optimized.get("execution_contract") or {}).get("initial_equity") or 10_000_000)
                result = run_olr_standalone(stressed.olr_config, stressed.olr_snapshots, stressed.olr_bars,
                    initial_equity=initial, capacity_policy=policy)
            else:
                result = run_native_risk_shared_replay(stressed, initial_equity=initial, capacity_policy=policy)
            print(json.dumps({"capacity_account": target, "scenario": name,
                "status": "replay_complete_auditing", "fills": len(result.broker.fills)}), flush=True)
            configuration = {**spec, "capacity_scenario": {"name": name, "participation": participation},
                "target": target, "initial_equity": initial, "extra_slippage_bps": extra,
                "consumed_market_bars": alignment["strategies"], "feed_scope": inputs.lineage["KALCB"]["feed_scope"]}
            request = stable_signature([configuration, active_manifest().semantic_identity()])
            audit = audit_broker(result.broker, identity={"request": request,
                "mutations": stable_signature([stressed.kalcb_config, stressed.olr_config]),
                "input_manifest": active_manifest().receipt(), "configuration": configuration,
                "lineage": inputs.lineage})
            record = {"summary": asdict(summarize_replay(result, initial_equity=initial)),
                "audit": audit, "request": request, "contract": policy.contract(),
                "open_positions": [asdict(p) for p in result.broker.positions.values()],
                "pending_orders": [asdict(o) for o in result.broker.orders]}
            output["accounts"][target][name] = record
            save()
            print(json.dumps({"capacity_account": target, "scenario": name,
                "status": audit["capacity_model_audit"]["status"], "summary": record["summary"]}), flush=True)
            del result
            gc.collect()
    output["inputs_unchanged_during_run"] = manifest_before == check_manifest()
    output["status"] = "completed_research_model_comparison" if output["inputs_unchanged_during_run"] else "invalid_input_generation_changed"
    save()
    print(json.dumps({"output": str(args.output), "status": output["status"]}), flush=True)


def run_single_diagnostic(args, inputs, spec, alignment, olr_optimized, manifest_before, check_manifest):
    """Fixed configuration diagnostics; release each account before the next run."""
    import gc
    from backtests.engine.integrity import audit_broker
    from backtests.auto.shared.cache_keys import stable_signature
    target = args.diagnostic_target
    from backtests.engine.capacity import capacity_from_config
    capacity_policy = capacity_from_config(spec)
    initial = float(spec["initial_equity"])
    if target == "KALCB":
        initial = inputs.kalcb_replay.initial_equity
        result = run_kalcb_standalone(inputs.kalcb_config, inputs.kalcb_replay, capacity_policy=capacity_policy)
    elif target == "OLR":
        initial = float((olr_optimized.get("execution_contract") or {}).get("initial_equity") or 10_000_000)
        result = run_olr_standalone(inputs.olr_config, inputs.olr_snapshots, inputs.olr_bars, initial_equity=initial, capacity_policy=capacity_policy)
    else:
        result = run_native_risk_shared_replay(inputs, initial_equity=initial, capacity_policy=capacity_policy)
    configuration = {**spec, "consumed_market_bars": alignment["strategies"],
        "feed_scope": inputs.lineage["KALCB"]["feed_scope"]}
    request = stable_signature([target, configuration, active_manifest().semantic_identity()])
    audit = audit_broker(result.broker, identity={"request": request, "input_manifest": active_manifest().receipt(),
        "configuration": configuration, "lineage": inputs.lineage})
    output = {"status": "diagnostic_only_unqualified", "historical_qualification": False,
        "target": target, "summary": asdict(summarize_replay(result, initial_equity=initial)),
        "audit": audit, "request": request, "input_alignment": alignment,
        "lineage": inputs.lineage, "input_manifest": manifest_before,
        "scope": "corrected_clock_fixed_configuration; legacy comparison retained in prior paired evidence",
        "open_positions": [asdict(p) for p in result.broker.positions.values()]}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
    del result
    gc.collect()
    if args.native_policy_grid:
        if target != "shared":
            raise ValueError("Native portfolio grid requires the shared target")
        from backtests.strategies.portfolio_synergy.native_evaluator import NativePortfolioPolicyEvaluator
        from backtests.auto.shared.types import Experiment
        evaluator = NativePortfolioPolicyEvaluator(inputs, initial_equity=initial, configuration=configuration)
        grid = json.loads(read_input_bytes(args.native_policy_grid))
        if not grid or len({row["name"] for row in grid}) != len(grid):
            raise ValueError("Native policy grid requires unique named experiments")
        output["native_policy_experiments"] = []
        for row in grid:
            candidate = evaluator([Experiment(row["name"], row["mutations"])], {})[0]
            output["native_policy_experiments"].append(asdict(candidate))
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
            gc.collect()
    output["inputs_unchanged_during_run"] = manifest_before == check_manifest()
    if not output["inputs_unchanged_during_run"]:
        output["status"] = "invalid_input_generation_changed"
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": output["status"], "summary": output["summary"]}))


def _olr_replay_pairs(
    snapshots: Mapping[date, OLRDailySnapshot],
    calendar: tuple[date, ...],
) -> set[tuple[date, str]]:
    calendar_index = {day: index for index, day in enumerate(calendar)}
    pairs: set[tuple[date, str]] = set()
    for day, snapshot in snapshots.items():
        if day not in calendar_index:
            raise ValueError(f"OLR snapshot date is outside the KALCB training calendar: {day}")
        offset = calendar_index[day]
        for candidate in snapshot.candidates:
            symbol = str(candidate.symbol).zfill(6)
            for replay_day in calendar[offset : min(offset + 3, len(calendar))]:
                pairs.add((replay_day, symbol))
    return pairs


def management_replay_pairs(snapshots, calendar):
    calendar = tuple(sorted(set(calendar)))
    first_seen = {}
    for day, snapshot in sorted(snapshots.items()):
        if day not in calendar:
            raise ValueError(f"Snapshot date is outside the replay calendar: {day}")
        for candidate in snapshot.candidates:
            first_seen.setdefault(str(candidate.symbol).zfill(6), day)
    return {(day, symbol) for symbol, first in first_seen.items() for day in calendar if day >= first}


def _load_pinned_bars(pairs: set[tuple[date, str]]) -> list[MarketBar]:
    contract = market_input_contract()
    return _load_pinned_parquet_bars(
        pairs, intraday_root=_repo_path(contract["data_root"] if contract else "data/kis_intraday_parquet"),
        data_snapshot_end=contract["snapshot_end"] if contract else PINNED_INTRADAY_END,
        snapshot_end_overrides=contract.get("snapshot_end_overrides") if contract else None,
    )


def _write_input_receipts(output, alignment, snapshots_by_strategy):
    """Keep new evidence beside the diagnostic; preserve original round artifacts."""
    from backtests.core.input_manifest import signature
    root = output.with_suffix(".inputs")
    receipts = {}
    for strategy, snapshots in snapshots_by_strategy.items():
        target = root / strategy.lower()
        target.mkdir(parents=True, exist_ok=False)
        snapshot_rows = [{"trade_date": day.isoformat(), "artifact_hash": snapshot.artifact_hash}
                         for day, snapshot in sorted(snapshots.items())]
        payloads = {
            "input_receipt.json": {key: value for key, value in alignment.items() if key != "strategies"},
            "replay_market_bar_digest.json": alignment["strategies"][strategy],
            "daily_snapshot_manifest.json": {"snapshot_count": len(snapshot_rows), "snapshots": snapshot_rows,
                                             "candidate_snapshot_hash": signature(snapshot_rows)},
        }
        for name, payload in payloads.items():
            (target / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        receipts[strategy] = str(target)
    return receipts


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(read_input_bytes(path).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _repo_path(raw: Any) -> Path:
    path = Path(str(raw))
    path = path if path.is_absolute() else ROOT / path
    return active_manifest().resolve(path) if active_manifest() else path


if __name__ == "__main__":
    main()
