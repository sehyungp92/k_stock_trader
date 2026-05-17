from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_yaml_config, normalize_runtime_config
from .strategies.registry import get_backtest_runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run K stock strategy backtests.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a strategy backtest.")
    run.add_argument("--strategy", required=True, choices=["kmp", "kpr", "nulrimok"])
    run.add_argument("--config", default=None)
    run.add_argument("--fixture", default="synthetic")
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = normalize_runtime_config(args.strategy, load_yaml_config(args.config))
    if args.fixture == "synthetic":
        config["capability_level"] = "synthetic"
    if args.dry_run:
        print(json.dumps({"strategy": args.strategy, "config": config, "dry_run": True}, indent=2, default=str))
        return 0
    result = get_backtest_runner(args.strategy)(config, {})
    print(json.dumps({"strategy": args.strategy, "metrics": result.metrics, "source_fingerprint": result.source_fingerprint}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

