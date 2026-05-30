from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.olr_kalcb.hashing import file_sha256

DEFAULT_BASELINE_MANIFEST = Path(
    os.environ.get("OLR_KALCB_BASELINE_MANIFEST", "data/live_readiness/olr_kalcb/2026-05-28/baseline_manifest.json")
)


def check_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("paper_trading_approved") is not False:
        failures.append({"path": str(manifest_path), "error": "paper_trading_approval_must_be_false"})
    if manifest.get("live_capital_approved") is not False:
        failures.append({"path": str(manifest_path), "error": "live_capital_approval_must_be_false"})
    seen_labels: set[str] = set()
    artifacts = list(manifest.get("artifacts", []))
    if not artifacts:
        failures.append({"path": str(manifest_path), "error": "no_artifacts"})
    for item in artifacts:
        label = str(item.get("label") or "")
        if not label:
            failures.append({"path": str(manifest_path), "error": "blank_artifact_label"})
        elif label in seen_labels:
            failures.append({"path": str(manifest_path), "error": "duplicate_artifact_label", "label": label})
        seen_labels.add(label)
        raw_path = str(item.get("path") or "")
        if not raw_path:
            failures.append({"path": str(manifest_path), "error": "blank_artifact_path", "label": label})
            continue
        target = Path(raw_path)
        if not target.exists():
            failures.append({"path": str(target), "error": "missing"})
            continue
        expected = str(item.get("sha256") or "").lower()
        if not expected:
            failures.append({"path": str(target), "error": "missing_sha256"})
            continue
        actual = file_sha256(target).lower()
        if actual != expected:
            failures.append({"path": str(target), "error": "sha256_mismatch", "expected": expected, "actual": actual})
    result = {
        "manifest": str(manifest_path),
        "checked": len(artifacts),
        "passed": not failures,
        "failures": failures,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an OLR/KALCB live-readiness baseline manifest.")
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_BASELINE_MANIFEST))
    args = parser.parse_args()
    result = check_manifest(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
