"""Revalidate saved execution labels from their authenticated fill evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timedelta
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def validate(source, output):
    from scripts.kalcb_executed_value_study import emit
    from backtests.auto.shared.cache_keys import source_generation_hash
    generation = source_generation_hash()
    original = json.loads((source / "labels_manifest.json").read_text())
    request = json.loads((source / "request.json").read_text())
    completion = json.loads((source / "completion.json").read_text())
    if not completion["complete"] or completion["source_generation_hash"] != request["source_generation_hash"] or ev.digest(source / "request.json") != original["request_sha256"]:
        raise ValueError("Original label generation incomplete or changed")
    output.mkdir(parents=True, exist_ok=False)
    ev.write_json(output / "request.json", {"operation": "independent_ex_post_fill_impossibility_validation",
        "source_generation_hash": generation, "original_manifest_path": str(source / "labels_manifest.json"),
        "original_manifest_sha256": ev.digest(source / "labels_manifest.json"), "original_request": request})
    counts = Counter()
    for name, expected in original["label_files"].items():
        if ev.digest(source / name) != expected:
            raise ValueError("Changed original labels")
        event_name = name.replace(".jsonl", ".events.jsonl.gz")
        if ev.digest(source / event_name) != original["event_files"][event_name]:
            raise ValueError("Changed original fill evidence")
        rows = {r["opportunity_id"]: r for r in ev.read_jsonl(source / name)}
        with gzip.open(source / event_name, "rt", encoding="utf-8") as events, (output / name).open("x", encoding="utf-8") as stream:
            for line in events:
                event = json.loads(line)
                row = rows.pop(event["opportunity_id"])
                failures = ev.physical_fill_failures(event["fills"])
                row["physical_fill_failures"] = failures
                if failures:
                    row.update(label_valid=False, status="execution_invalid", net_value_krw=None,
                               net_value_equity_bps=None, label_available_at=None)
                    row["risk_available_at"] = max(datetime.fromisoformat(row["risk_available_at"]),
                        *(datetime.fromisoformat(f["timestamp"]) + timedelta(minutes=5) for f in failures)).isoformat()
                counts[row["status"]] += 1
                counts["deadline_violations"] += row["deadline_violation"]
                emit(stream, row)
        if rows:
            raise ValueError("Missing execution details")
        try:
            os.link(source / event_name, output / event_name)
        except OSError:
            shutil.copy2(source / event_name, output / event_name)
    ev.write_json(output / "labels_manifest.json", {**original, "counts": counts,
        "request_sha256": ev.digest(output / "request.json"),
        "original_manifest_sha256": ev.digest(source / "labels_manifest.json"),
        "label_files": {p.name: ev.digest(p) for p in output.glob("*.jsonl")}})
    if source_generation_hash() != generation:
        raise RuntimeError("Validator source changed")
    ev.write_json(output / "completion.json", {"complete": True, "source_generation_hash": generation})
    print(json.dumps({"validated": str(output), "counts": counts}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    validate(a.source, a.output)
