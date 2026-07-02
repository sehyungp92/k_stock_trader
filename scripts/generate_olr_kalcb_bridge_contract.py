from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from deployment.olr_kalcb.bridge_contract import write_strategy_plugin_contract
from deployment.olr_kalcb.deployment_metadata import DEFAULT_CONTRACT_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the checked-in OLR/KALCB assistant bridge contract.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CONTRACT_PATH),
        help="Contract JSON output path.",
    )
    args = parser.parse_args(argv)
    contract = write_strategy_plugin_contract(args.output, repo_root=REPO_ROOT)
    print(
        json.dumps(
            {
                "path": str((REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)),
                "schema_version": contract["schema_version"],
                "contract_hash": contract["contract_hash"],
                "strategy_ids": contract["strategy_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
