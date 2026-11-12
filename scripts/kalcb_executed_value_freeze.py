"""Bind a derived chronological candidate policy to the unchanged research auditor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def freeze(parent_path, code, policies, prepared, labels, plan, output, additional_evidence=()):
    parent = json.loads(parent_path.read_text())
    identity = parent.pop("manifest_hash")
    if signature(parent) != identity:
        raise ValueError("Invalid parent manifest")
    output.mkdir(parents=True, exist_ok=False)
    root = output / "files"
    files = {}
    def bind(source, relative, expected=None):
        source = Path(source)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Duplicate immutable binding: {relative}")
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        files[Path(relative).as_posix()] = expected or digest(target)
    for name, value in parent["files"].items():
        replacement = code / name
        if replacement.exists() and (name.endswith(".py") or name.startswith("kis_core/data/")):
            bind(replacement, name)
        else:
            bind(parent_path.parent / "files" / name, name, value)
    for directory in ("backtests", "strategy_common", "strategy_kalcb", "strategy_olr", "strategy_pcim", "deployment", "instrumentation", "oms", "oms_client", "kis_core", "scripts"):
        for path in (code / directory).rglob("*.py"):
            name = path.relative_to(code).as_posix()
            if name not in files:
                bind(path, name)
    policy_prefix = Path("research/kalcb_executed_value/policies")
    for path in policies.iterdir():
        if path.is_file():
            bind(path, policy_prefix / path.name)
    for path in prepared.iterdir():
        if path.is_file():
            bind(path, Path("research/kalcb_executed_value/prepared") / path.name)
    label_prefixes = []
    for index, directory in enumerate(labels):
        prefix = Path(f"research/kalcb_executed_value/labels_{index}")
        label_prefixes.append(prefix.as_posix())
        for path in directory.iterdir():
            if path.is_file():
                bind(path, prefix / path.name)
    bind(plan, "research/kalcb_executed_value/plan.json")
    bind(Path(__file__), "research/kalcb_executed_value/freeze_program.py")
    extra_prefixes = []
    for index, directory in enumerate(additional_evidence):
        prefix = Path(f"research/kalcb_executed_value/additional_evidence_{index}")
        extra_prefixes.append({"source": str(directory), "bound": prefix.as_posix()})
        for path in directory.rglob("*"):
            if path.is_file():
                bind(path, prefix / path.relative_to(directory))
    # Bind inputs outside the parent bundle too, including the exact database and
    # archived target/guard sources. Relative hash-prefixed names avoid collisions.
    request = json.loads(plan.read_text())
    for path, expected in request["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Changed declared study input")
        bind(path, f"research/kalcb_executed_value/inputs/{expected[:12]}_{Path(path).name}", expected)
    program = "import sys;sys.path.insert(0,sys.argv[1]);from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())"
    source_hash = subprocess.check_output([sys.executable, "-B", "-c", program, str(root)], text=True).strip()
    evidence = dict(parent["evidence"])
    evidence["executed_value_policy"] = {"policies_manifest": (policy_prefix / "policies_manifest.json").as_posix(),
        "prepared": "research/kalcb_executed_value/prepared/prepared.json", "label_directories": label_prefixes,
        "additional_evidence": extra_prefixes,
        "plan": "research/kalcb_executed_value/plan.json", "parent_manifest_hash": identity}
    payload = {"version": parent["version"], "source_root": parent["source_root"],
               "source_generation_hash": source_hash, "files": files, "evidence": evidence}
    write_json(output / "manifest.json", {**payload, "manifest_hash": signature(payload)})
    print(json.dumps({"manifest": str(output / "manifest.json"), "files": len(files), "source_generation_hash": source_hash}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "code", "policies", "prepared", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--labels", type=Path, action="append", required=True)
    p.add_argument("--evidence", type=Path, action="append", default=[])
    a = p.parse_args()
    freeze(a.parent, a.code, a.policies, a.prepared, a.labels, a.plan, a.output, a.evidence)


if __name__ == "__main__":
    main()
