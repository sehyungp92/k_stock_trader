"""Regenerate OLR final artifacts through RoundManager and PhaseRunner.

Select the target explicitly with --round. The historical round5 script delegates
here and retains its default only for compatibility with recorded commands.
The default path replays natively. Export-only reporting requires sealed files
and passing receipts bound to the exact source, inputs, recipe and scenario.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None, *, default_round=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=default_round, required=default_round is None)
    parser.add_argument("--config", type=Path, default=ROOT / "config/optimization/olr.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/backtests/output")
    parser.add_argument("--execution-root", type=Path,
                        help="Frozen execution checkout used by the native exports")
    parser.add_argument("--exported-cases", type=Path,
                        help="Sealed native diagnostic case plan; revalidate every receipt and export before reporting")
    parser.add_argument("--native-capacity-cases", type=Path,
                        help="Registered native capacity study cases, preserving their original feed and receipt schema")
    parser.add_argument("--native-interval-cases", type=Path,
                        help="Sealed retained I0/N0 cases; validate each window under its own frozen engine")
    parser.add_argument('--verified-interval-report', type=Path,
                        help='Preserved completed native report to reuse for reporting-only analysis')
    parser.add_argument('--verified-interval-report-sha256',
                        help='Required exact parent JSON digest; no unsealed report cache')
    parser.add_argument("--baseline-from-config", action="store_true",
                        help="Adopt the explicitly configured baseline; no optimization phases are claimed")
    args = parser.parse_args(argv)
    if (args.verified_interval_report or args.verified_interval_report_sha256) and not (
            args.native_interval_cases and args.verified_interval_report and args.verified_interval_report_sha256):
        parser.error('Verified report reuse requires interval cases, parent directory and explicit SHA-256')
    if sum(bool(v) for v in (args.exported_cases, args.native_capacity_cases, args.native_interval_cases)) > 1:
        parser.error("Choose one native export schema")
    if args.native_interval_cases and args.execution_root:
        parser.error("Interval cases bind their own per-window engine roots; omit --execution-root")
    if args.round < 1:
        parser.error("--round must be positive")
    return parser, args


def main(argv=None, *, default_round=None):
    parser, args = parse_args(argv, default_round=default_round)
    # Select the immutable execution checkout before importing its validators.
    # Parsing explicit argv here also keeps help and invalid requests side-effect free.
    sys.path.insert(0, str(args.execution_root or ROOT))
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.phase_state import PhaseState, save_phase_state
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.config import load_yaml_config
    from backtests.strategies.olr.plugin import OLROptimizationPlugin
    from backtests.strategies.olr.final_diagnostics import require_bound_result

    config = load_yaml_config(args.config)
    if config.get("native_case_plan_sha256") and not (args.native_capacity_cases or args.native_interval_cases):
        parser.error("This saved recipe binds a native case plan. Supply its matching --native-capacity-cases or --native-interval-cases adapter; the generic OLR feed is not equivalent.")
    manager = RoundManager("stock", "olr", base_dir=args.output_root)
    directory = manager.get_round_dir(args.round)
    directory.mkdir(parents=True, exist_ok=True)
    if args.baseline_from_config:
        mutations = dict(config["initial_mutations"])
    else:
        mutations = json.loads((directory / "optimized_config.json").read_text(encoding="utf-8"))["mutations"]
        config["initial_mutations"] = mutations
    if args.native_interval_cases:
        from backtests.strategies.olr.interval_round_diagnostics import IntervalDiagnosticsPlugin
        plugin = IntervalDiagnosticsPlugin(config, args.native_interval_cases, output_dir=directory)
        if args.verified_interval_report:
            plugin.reuse_verified_report(args.verified_interval_report, args.verified_interval_report_sha256)
    elif args.native_capacity_cases:
        spec = importlib.util.spec_from_file_location("backtests.strategies.olr.native_round_diagnostics",
            ROOT / "backtests/strategies/olr/native_round_diagnostics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        plugin = module.NativeCapacityDiagnosticsPlugin(config, args.native_capacity_cases, output_dir=directory)
    elif args.exported_cases:
        spec = importlib.util.spec_from_file_location("backtests.strategies.olr.exported_diagnostics",
            ROOT / "backtests/strategies/olr/exported_diagnostics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(f"Revalidating sealed native exports for OLR round {args.round}", flush=True)
        plugin = module.ExportedOLRDiagnosticsPlugin(config, args.exported_cases, output_dir=directory)
    else:
        print(f"Rebuilding OLR round {args.round} from current source and frozen inputs", flush=True)
        plugin = OLROptimizationPlugin(config, output_dir=directory, max_workers=1)
    if args.native_capacity_cases or args.native_interval_cases:
        # Capacity receipts bind the frozen study/recipe/scenario rather than
        # the generic runner's config/mutations identity. Never rewrite them.
        plugin.require_bound_cases(mutations)
    else:
        require_bound_result(plugin._evaluate_result(mutations), plugin.training_config, mutations)
    runner = PhaseRunner(plugin, directory, config.get("baseline_name", "OLR validated baseline"),
                         round_manager=manager, round_num=args.round)
    state = PhaseState(cumulative_mutations=mutations, round_name=runner.round_name) if args.baseline_from_config else runner.load_state()
    state.cumulative_mutations = mutations
    manager.write_run_spec(directory, args.round, "olr", baseline_mutations=mutations,
        baseline_source=args.config, baseline_metadata={"adoption": args.baseline_from_config,
        "consumed_regression_not_untouched_oos": True}, execution_context=plugin.execution_context,
        description=f"{runner.round_name}; saved-recipe adoption and full diagnostics" if args.baseline_from_config else "Current-source final diagnostics refresh",
        overwrite=True)
    save_phase_state(state, runner.state_path)
    runner.run_end_of_round(state)
    print(f"Completed full OLR diagnostics: {directory}", flush=True)


if __name__ == "__main__":
    main()
