"""Collect fresh KIS paper evidence without submitting or cancelling orders."""
import argparse
import asyncio
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def collect(args):
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    required = ("KIS_PAPER_APP_KEY", "KIS_PAPER_APP_SECRET", "KIS_PAPER_ACCOUNT_NO")
    if os.environ.get("KIS_IS_PAPER", "").lower() != "true":
        raise ValueError("Set KIS_IS_PAPER=true explicitly for this paper-only collector")
    if any(not os.environ.get(key) for key in required):
        raise ValueError("Explicit KIS_PAPER_* credentials and account are required")
    from kis_core.kis_auth import KoreaInvestEnv, build_kis_config_from_env
    from kis_core.kis_client import KoreaInvestAPI
    from oms.adapter import KISExecutionAdapter
    from oms.broker_snapshot import collect_broker_snapshot
    cfg = build_kis_config_from_env()
    # This diagnostic needs no live market-data fallback.
    for key in ("url", "api_key", "api_secret_key"):
        cfg.pop(key, None)
    api = await asyncio.to_thread(lambda: KoreaInvestAPI(KoreaInvestEnv(cfg)))
    return await collect_broker_snapshot(KISExecutionAdapter(api), oms_id=args.oms_id,
                                        account_alias=args.account_alias, history_start=args.history_start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--oms-id", required=True)
    parser.add_argument("--account-alias", required=True)
    parser.add_argument("--history-start", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve evidence")
    try:
        packet = asyncio.run(collect(args))
        passed = True
    except Exception as exc:
        # Avoid dumping HTTP payloads, secrets or account identifiers on failure.
        packet = {"queries_complete": False, "mode": "paper", "oms_id": args.oms_id,
                  "account_alias": args.account_alias, "as_of": datetime.now(timezone.utc).isoformat(),
                  "failure_type": type(exc).__name__, "paper_ready": False}
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(packet, output, indent=2, default=str)
        output.write("\n")
    print(json.dumps({"queries_complete": passed, "output": str(args.output.resolve()),
                      "failure_type": packet.get("failure_type"), "orders_submitted": 0}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
