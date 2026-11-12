"""Inspect canonical execution state against a supplied broker snapshot without writes."""
import argparse
import asyncio
import json
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asyncpg
from oms.persistence import OMSPersistence
from oms.execution_preflight import execution_preflight


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oms-id", required=True)
    parser.add_argument("--broker-snapshot", required=True, type=Path)
    parser.add_argument("--risk-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="STRATEGY=PATH",
                        help="Current durable runtime checkpoint; repeat for each enabled strategy")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve cutover evidence")
    snapshot = json.loads(args.broker_snapshot.read_text(encoding="utf-8"))
    risk = json.loads(args.risk_contract.read_text(encoding="utf-8"))
    checkpoints = []
    for spec in args.checkpoint:
        strategy, separator, path = spec.partition("=")
        if not separator:
            parser.error("--checkpoint requires STRATEGY=PATH")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("checkpoint_version") != "canonical-execution-v2" or payload.get("pending_event"):
            parser.error("Checkpoint must be canonical and have no pending feedback event")
        checkpoints.append({"strategy_id": strategy.upper(), "meta": payload["state"]["meta"]})
    if snapshot.get("oms_id") != args.oms_id or not snapshot.get("as_of"):
        parser.error("Snapshot requires the matching oms_id and an as_of timestamp")
    if not os.environ.get("DATABASE_URL"):
        parser.error("DATABASE_URL must explicitly identify the OMS database")
    store = OMSPersistence(dsn=os.environ["DATABASE_URL"], oms_id=args.oms_id)
    store.pool = await asyncpg.create_pool(store.dsn, min_size=1, max_size=1,
        server_settings={"default_transaction_read_only": "on"})
    try:
        report = await execution_preflight(store, snapshot["positions"], checkpoints=checkpoints or snapshot.get("checkpoints", []),
                                           broker_snapshot=snapshot, risk_contract=risk)
        report["broker_snapshot_as_of"] = snapshot["as_of"]
        report["broker_snapshot_sha256"] = hashlib.sha256(args.broker_snapshot.read_bytes()).hexdigest()
        report["oms_id"] = args.oms_id
        report["account_fingerprint"] = snapshot.get("account_fingerprint")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    finally:
        await store.pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
