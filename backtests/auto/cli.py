from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.registry import create_plugin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phased auto-optimisation.")
    sub = parser.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize", help="Run phased greedy optimisation.")
    optimize.add_argument("--strategy", required=True, choices=["kmp", "kpr", "nulrimok"])
    optimize.add_argument("--config", default=None)
    optimize.add_argument("--round-name", default="round")
    optimize.add_argument("--round", type=int, default=None)
    optimize.add_argument("--output-root", default="data/backtests/output")
    optimize.add_argument("--max-workers", type=int, default=1)
    optimize.add_argument("--num-phases", type=int, default=None)
    optimize.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = normalize_runtime_config(args.strategy, load_yaml_config(args.config))
    manager = RoundManager("stock", args.strategy, base_dir=Path(args.output_root))
    if args.dry_run:
        latest = manager.get_latest_round()
        round_num = args.round or (latest + 1 if latest else 1)
        round_dir = manager.round_path(round_num)
        print(json.dumps({"strategy": args.strategy, "round": round_num, "round_dir": str(round_dir), "dry_run": True}, indent=2))
        return 0
    round_num, round_dir = manager.resolve_round(args.round, for_write=True, expected_phases=args.num_phases)
    plugin = create_plugin(args.strategy, config, output_dir=round_dir, max_workers=args.max_workers, capability_level=config.get("capability_level", "synthetic"))
    if args.num_phases is not None:
        plugin.num_phases = int(args.num_phases)
    runner = PhaseRunner(plugin, round_dir, round_name=args.round_name, round_manager=manager, round_num=round_num)
    state = runner.run_all_phases()
    print(json.dumps({"strategy": args.strategy, "round": round_num, "completed_phases": state.completed_phases, "round_dir": str(round_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
