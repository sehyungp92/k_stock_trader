from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.olr_kalcb.offline_replay import rebuild_offline_replay_from_session
from deployment.olr_kalcb.replay import replay_paper_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay and hash-check an OLR/KALCB paper session bundle.")
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--hash-contract-only",
        action="store_true",
        help="Debug-only: print hash-contract status without making it promotional or successful without replay.",
    )
    parser.add_argument(
        "--allow-debug-success",
        action="store_true",
        help="With --hash-contract-only only, return success for a complete matching hash contract. Non-promotional debugging only.",
    )
    parser.add_argument(
        "--build-offline-replay",
        action="store_true",
        help="Regenerate offline_replay/ from captured artifacts and 5m bars before checking parity.",
    )
    args = parser.parse_args()
    if args.build_offline_replay:
        rebuild_offline_replay_from_session(args.session)
    report = replay_paper_session(args.session)
    print(
        json.dumps(
            {
                "session": report["session"],
                "replay_mode": report["replay_mode"],
                "behavior_parity_passed": report["behavior_parity_passed"],
                "paper_gate_passed": report["paper_gate_passed"],
                "paper_gate_status": report["paper_gate_status"],
                "promotion_blockers": report["promotion_blockers"],
                "hash_contract_passed": report["hash_contract_passed"],
                "session_bundle_complete": report["session_bundle_complete"],
                "mismatches": len(report["mismatches"]),
            },
            indent=2,
        )
    )
    hash_contract_ok = report["hash_contract_passed"] and report["session_bundle_complete"]
    debug_success = args.hash_contract_only and args.allow_debug_success and hash_contract_ok
    return 0 if report["behavior_parity_passed"] or debug_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
